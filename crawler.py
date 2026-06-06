"""Auto-archiver: walks Civitai newest-first for the configured base models and
archives any LoRA version not yet in the manifest. Intended to run on a schedule
(GitHub Actions cron).

MODE=new: archives only versions published within the last SINCE_DAYS.
MODE=backfill: walks the catalog in descending publish-date windows
(WINDOW_DAYS at a time), sorting within each window so the archive fills
strictly newest-first by publish date. State persists in backfill_state.json
so subsequent runs continue from where the last one stopped.

Files are downloaded to a staging dir and uploaded in BATCHED commits (one
commit per batch via upload_folder) to stay under HF's 128-commits/hour cap.
Streams Civitai -> runner -> HF; nothing persists on the runner."""
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone

from huggingface_hub import HfApi

from core import (
    MANIFEST_PATH,
    iter_lora_versions,
    load_backfill_ceiling,
    load_manifest,
    load_skiplist,
    parse_dt,
    pick_primary_file,
    save_backfill_ceiling,
    save_skiplist,
    stage_download,
    upload_folder_with_retry,
)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
ARCHIVE_REPO = os.environ.get("ARCHIVE_REPO", "isam/civitai-lora-archive")
BASE_MODELS = [b.strip() for b in os.environ.get("BASE_MODELS", "Flux.1 D").split(",") if b.strip()]
MODE = os.environ.get("MODE", "new").strip().lower()
SINCE_DAYS = float(os.environ.get("SINCE_DAYS", "2"))  # 'new' lookback window
WINDOW_DAYS = float(os.environ.get("WINDOW_DAYS", "7"))  # backfill date-window size
MIN_CEILING = parse_dt(os.environ.get("BACKFILL_FLOOR", "2018-01-01T00:00:00Z"))
MAX_EMPTY_WINDOWS = int(os.environ.get("MAX_EMPTY_WINDOWS", "5"))  # stop after N consecutive empty
MAX_FILES_PER_RUN = int(os.environ.get("MAX_FILES_PER_RUN", "0"))  # 0 = unlimited
MAX_FILE_MB = int(os.environ.get("MAX_FILE_MB", "0"))  # 0 = no limit
BATCH_FILES = int(os.environ.get("BATCH_FILES", "40"))
BATCH_GB = float(os.environ.get("BATCH_GB", "8"))
BATCH_BYTES = BATCH_GB * (1024 ** 3)

PERMANENT_HTTP = {401, 403, 404}


class Ctx:
    """Mutable per-run state shared between the staging loop and the flush helper."""
    def __init__(self, api, archived, skiplist):
        self.api = api
        self.archived = archived
        self.skiplist = skiplist
        self.skiplist_dirty = False
        self.staging = tempfile.mkdtemp(prefix="civitai_batch_")
        self.batch_files = 0
        self.batch_bytes = 0
        self.batch_idx = 0
        self.new_count = 0
        self.skipped_size = 0

    def flush(self):
        if self.batch_files == 0:
            return
        with open(os.path.join(self.staging, MANIFEST_PATH), "w") as fh:
            json.dump({"archived_version_ids": sorted(self.archived, key=lambda x: int(x))}, fh, indent=2)
        self.batch_idx += 1
        print(f"  -> flushing batch {self.batch_idx}: {self.batch_files} files / {self.batch_bytes/(1024**3):.2f}GB", flush=True)
        upload_folder_with_retry(
            self.api, ARCHIVE_REPO, self.staging, f"Archive batch {self.batch_idx} ({self.batch_files} files)"
        )
        shutil.rmtree(self.staging, ignore_errors=True)
        self.staging = tempfile.mkdtemp(prefix="civitai_batch_")
        self.batch_files = 0
        self.batch_bytes = 0

    def cleanup(self):
        shutil.rmtree(self.staging, ignore_errors=True)


def process_version(ctx, model, version):
    """Attempt to archive one (model, version). Returns:
       'ok' staged, 'dup' skipped (manifest/skip), 'skip' size/no-file, 'cap' cap hit, 'err' fail."""
    vid = str(version.get("id"))
    if vid in ctx.archived or vid in ctx.skiplist:
        return "dup"
    if MAX_FILES_PER_RUN > 0 and ctx.new_count >= MAX_FILES_PER_RUN:
        return "cap"

    primary = pick_primary_file(version)
    if not primary:
        return "skip"
    size_mb = (primary.get("sizeKB", 0) or 0) / 1024
    if MAX_FILE_MB and size_mb > MAX_FILE_MB:
        print(f"  skip (>{MAX_FILE_MB}MB): {model.get('name')} v{vid} ({size_mb:.0f}MB)")
        ctx.skipped_size += 1
        return "skip"

    try:
        result = stage_download(ctx.staging, model, version)
    except Exception as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in PERMANENT_HTTP:
            ctx.skiplist.add(vid)
            ctx.skiplist_dirty = True
            print(f"  SKIP (HTTP {status}, won't retry): {model.get('name')} v{vid}")
        else:
            print(f"  FAIL {model.get('name')} v{vid}: {e}", file=sys.stderr)
        return "err"
    if not result:
        return "skip"
    rel_path, got = result
    ctx.archived.add(vid)
    ctx.new_count += 1
    ctx.batch_files += 1
    ctx.batch_bytes += got
    print(f"  [{ctx.new_count}] staged {rel_path} ({got/(1024**2):.0f}MB)", flush=True)
    if ctx.batch_files >= BATCH_FILES or ctx.batch_bytes >= BATCH_BYTES:
        ctx.flush()
    return "ok"


def run_new(ctx, base_set):
    published_after = datetime.now(timezone.utc) - timedelta(days=SINCE_DAYS)
    print(f"Mode: new (published since {published_after:%Y-%m-%d %H:%M}Z, last {SINCE_DAYS}d)")
    for model, version in iter_lora_versions(base_set, published_after=published_after, verbose=True):
        outcome = process_version(ctx, model, version)
        if outcome == "cap":
            print(f"Hit per-run cap ({MAX_FILES_PER_RUN}); rest picked up next run.")
            break
    ctx.flush()


def run_heal(ctx, base_set):
    """Single newest-first pass from now back to the backfill ceiling.
    Dedupe (manifest + skiplist) keeps it cheap — only true gaps download.
    Catches versions silently dropped on transient errors during normal runs."""
    floor = load_backfill_ceiling(ctx.api, ARCHIVE_REPO) or (
        datetime.now(timezone.utc) - timedelta(days=365)
    )
    print(f"Mode: heal (newest -> {floor:%Y-%m-%d %H:%M}Z, scanning for gaps in manifest+skiplist)")
    for model, version in iter_lora_versions(base_set, published_after=floor, verbose=True):
        outcome = process_version(ctx, model, version)
        if outcome == "cap":
            print(f"Hit per-run cap ({MAX_FILES_PER_RUN}); rest picked up next heal run.")
            break
    ctx.flush()


def run_backfill(ctx, base_set):
    ceiling = load_backfill_ceiling(ctx.api, ARCHIVE_REPO) or datetime.now(timezone.utc)
    # Track the deepest publishedAt we've actually examined (called process_version on).
    # The window-top `ceiling` only advances on clean window completion, so it lags reality
    # badly when a run is killed by timeout. `deepest` is monotonic-descending and reflects
    # actual scan progress, so persisting it survives mid-window kills without losing work.
    deepest = ceiling
    print(f"Mode: backfill (descending date windows of {WINDOW_DAYS}d, starting from {ceiling:%Y-%m-%d %H:%M}Z)")
    empty_streak = 0
    cap_hit = False
    try:
        while not cap_hit and empty_streak < MAX_EMPTY_WINDOWS and ceiling > MIN_CEILING:
            floor = ceiling - timedelta(days=WINDOW_DAYS)
            print(f"\n=== Window [{floor:%Y-%m-%d %H:%M} -> {ceiling:%Y-%m-%d %H:%M}] ===", flush=True)
            # Collect every qualifying version in [floor, ceiling); iter early-stops on cursor < floor.
            window = []
            for model, version in iter_lora_versions(base_set, published_after=floor, verbose=True):
                pub = parse_dt(version.get("publishedAt"))
                if pub is None or pub >= ceiling or pub < floor:
                    continue
                window.append((pub, model, version))
            window.sort(key=lambda t: t[0], reverse=True)
            print(f"  window has {len(window)} candidate versions", flush=True)
            if not window:
                empty_streak += 1
            else:
                empty_streak = 0
            for pub, model, version in window:
                batch_before = ctx.batch_files
                outcome = process_version(ctx, model, version)
                # Every version we just called through process_version (regardless of outcome)
                # has been examined. Versions are sorted desc by pub, so `pub` is monotonic-desc.
                if pub < deepest:
                    deepest = pub
                # A drop from batch_before>0 to 0 means process_version triggered ctx.flush().
                # Persist scan progress at this natural batch boundary so a SIGTERM after this
                # point still records the depth we reached.
                if outcome == "ok" and batch_before > 0 and ctx.batch_files == 0:
                    save_backfill_ceiling(ctx.api, ARCHIVE_REPO, deepest)
                    print(f"  ceiling persisted at {deepest:%Y-%m-%d %H:%M}Z", flush=True)
                if outcome == "cap":
                    cap_hit = True
                    print(f"Hit per-run cap ({MAX_FILES_PER_RUN}); resuming this window next run.")
                    break
            if cap_hit:
                # Don't advance ceiling — re-process this window next run (dedupe skips done items).
                break
            ceiling = floor  # window fully processed; descend
            if floor < deepest:
                deepest = floor  # record that we've examined everything down to this floor
            # Persist eagerly per-window. SIGTERM during the next window's pagination
            # kills the process inside a network call before any `finally` runs, so
            # in-memory state without a write here is lost on cancellation.
            save_backfill_ceiling(ctx.api, ARCHIVE_REPO, deepest)
            print(f"  ceiling persisted at {deepest:%Y-%m-%d %H:%M}Z", flush=True)
    finally:
        ctx.flush()
        # Save the actual scan depth, not the window-top ceiling that may not have advanced.
        final = min(ceiling, deepest)
        save_backfill_ceiling(ctx.api, ARCHIVE_REPO, final)
        print(f"Backfill ceiling saved at {final:%Y-%m-%d %H:%M}Z")


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set", file=sys.stderr)
        return 1

    base_set = set(BASE_MODELS)
    cap_desc = MAX_FILES_PER_RUN if MAX_FILES_PER_RUN > 0 else "unlimited"
    print(f"Base models: {sorted(base_set)}")
    print(f"Archive repo: {ARCHIVE_REPO}  |  per-run cap: {cap_desc}  |  batch: {BATCH_FILES} files / {BATCH_GB}GB")

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=ARCHIVE_REPO, repo_type="model", exist_ok=True)
    archived = load_manifest(api, ARCHIVE_REPO)
    skiplist = load_skiplist(api, ARCHIVE_REPO)
    print(f"Manifest: {len(archived)} archived  |  skiplist: {len(skiplist)} permanent failures")

    ctx = Ctx(api, archived, skiplist)
    try:
        if MODE == "backfill":
            run_backfill(ctx, base_set)
        elif MODE == "heal":
            run_heal(ctx, base_set)
        else:
            run_new(ctx, base_set)
    finally:
        ctx.cleanup()

    if ctx.skiplist_dirty:
        save_skiplist(api, ARCHIVE_REPO, skiplist)
        print(f"Skiplist updated: {len(skiplist)} permanent failures")
    print(f"Done. Archived {ctx.new_count} new file(s) this run; {ctx.skipped_size} skipped on size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
