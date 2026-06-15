"""Auto-archiver for Flux 1 Dev *checkpoints* (the LoRA crawler's big sibling).

Walks Civitai newest-first for Checkpoint models matching the configured base
models and archives every weight variant (full fp16, fp8, pruned — all files
tagged Civitai type "Model") not yet in the manifest. Targets the secondary
HF account, which has a finite storage budget, so the run STOPS once the next
file would push the repo past STORAGE_LIMIT_TB.

Differences from crawler.py (LoRAs):
  * types=Checkpoint, and ALL weight files per version (not just the primary).
  * Files are 6-24GB each, so there is NO batching: each file is downloaded,
    uploaded in its own commit, then deleted — peak disk stays at one file.
    State (manifest + storage tally + skiplist) rides along in that same commit,
    so it is atomic and resumable at file granularity.
  * A hard storage ceiling (STORAGE_LIMIT_TB) gates every download.

MODE=new: archive versions published within the last SINCE_DAYS.
MODE=backfill: descend the catalog in WINDOW_DAYS windows, newest-first,
persisting a ceiling so reruns continue where they stopped.
MODE=heal: one newest->ceiling pass that only fills manifest/skiplist gaps.

Streams Civitai -> runner -> HF; nothing persists on the runner."""
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone

from huggingface_hub import HfApi

from core import (
    iter_lora_versions,
    load_backfill_ceiling,
    load_repo_json,
    model_weight_files,
    parse_dt,
    save_backfill_ceiling,
    stage_file,
    upload_folder_with_retry,
)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
ARCHIVE_REPO = os.environ.get("ARCHIVE_REPO", "civitai2026/flux1-dev-checkpoints")
BASE_MODELS = [b.strip() for b in os.environ.get("BASE_MODELS", "Flux.1 D").split(",") if b.strip()]
MODEL_TYPE = os.environ.get("MODEL_TYPE", "Checkpoint").strip()
# Which Civitai file `type`s to grab. "Model" = the weight files (every fp/precision
# variant). Set FILE_TYPES=all to also pull VAE/Config/Training Data.
_ft = os.environ.get("FILE_TYPES", "Model").strip()
FILE_TYPES = None if _ft.lower() == "all" else tuple(t.strip() for t in _ft.split(",") if t.strip())

MODE = os.environ.get("MODE", "new").strip().lower()
SINCE_DAYS = float(os.environ.get("SINCE_DAYS", "3"))
WINDOW_DAYS = float(os.environ.get("WINDOW_DAYS", "30"))
MIN_CEILING = parse_dt(os.environ.get("BACKFILL_FLOOR", "2024-07-01T00:00:00Z"))  # Flux.1 launch
MAX_EMPTY_WINDOWS = int(os.environ.get("MAX_EMPTY_WINDOWS", "6"))
MAX_FILES_PER_RUN = int(os.environ.get("MAX_FILES_PER_RUN", "0"))  # 0 = unlimited
MAX_FILE_MB = int(os.environ.get("MAX_FILE_MB", "0"))  # 0 = no per-file size cap
STORAGE_LIMIT_TB = float(os.environ.get("STORAGE_LIMIT_TB", "8.5"))  # under the 8.6TB quota
STORAGE_LIMIT_BYTES = int(STORAGE_LIMIT_TB * (1024 ** 4))
STAGING_DIR = os.environ.get("STAGING_DIR") or None  # workflow points this at /mnt

MANIFEST_PATH = "manifest.json"
SKIPLIST_PATH = "skiplist.json"
STORAGE_PATH = "storage_state.json"
PERMANENT_HTTP = {401, 403, 404}


def _gb(b):
    return b / (1024 ** 3)


def _tb(b):
    return b / (1024 ** 4)


class Ctx:
    """Per-run state. One staging dir per file; state files ride in each commit."""
    def __init__(self, api, archived, skiplist, used_bytes):
        self.api = api
        self.archived = archived          # set of archived Civitai file IDs (str)
        self.skiplist = skiplist          # set of permanently-failed file IDs (str)
        self.skiplist_dirty = False
        self.used_bytes = used_bytes      # bytes already in the repo
        self.new_count = 0
        self.new_bytes = 0
        self.skipped_size = 0
        self.budget_full = False

    def _state_payload(self):
        """The three state files, as (filename, bytes) pairs, reflecting current memory."""
        return {
            MANIFEST_PATH: json.dumps(
                {"archived_file_ids": sorted(self.archived, key=int)}, indent=2
            ).encode(),
            STORAGE_PATH: json.dumps(
                {"used_bytes": self.used_bytes, "used_tb": round(_tb(self.used_bytes), 4),
                 "limit_tb": STORAGE_LIMIT_TB}, indent=2
            ).encode(),
            SKIPLIST_PATH: json.dumps(
                {"failed_file_ids": sorted(self.skiplist, key=int)}, indent=2
            ).encode(),
        }

    def save_skiplist_standalone(self):
        """Persist the skiplist on its own — for runs that only hit failures and
        thus never rode it along in a file commit."""
        self.api.upload_file(
            path_or_fileobj=json.dumps(
                {"failed_file_ids": sorted(self.skiplist, key=int)}, indent=2
            ).encode(),
            path_in_repo=SKIPLIST_PATH,
            repo_id=ARCHIVE_REPO,
            repo_type="model",
            commit_message=f"Update skiplist ({len(self.skiplist)} permanent failures)",
        )


def archive_file(ctx, model, version, fileobj):
    """Download+upload one weight file in its own commit. Returns outcome string:
       'ok' | 'dup' | 'skip' | 'cap' | 'full' | 'err'."""
    fid = str(fileobj.get("id"))
    if fid in ctx.archived or fid in ctx.skiplist:
        return "dup"
    if MAX_FILES_PER_RUN > 0 and ctx.new_count >= MAX_FILES_PER_RUN:
        return "cap"

    size = int(fileobj.get("sizeKB", 0) or 0) * 1024
    size_mb = size / (1024 ** 2)
    if MAX_FILE_MB and size_mb > MAX_FILE_MB:
        print(f"  skip (>{MAX_FILE_MB}MB): {model.get('name')} f{fid} ({size_mb:.0f}MB)")
        ctx.skipped_size += 1
        return "skip"
    if ctx.used_bytes + size > STORAGE_LIMIT_BYTES:
        print(f"  STORAGE FULL: used {_tb(ctx.used_bytes):.3f}TB + {_gb(size):.1f}GB "
              f"would exceed {STORAGE_LIMIT_TB}TB cap; stopping.", flush=True)
        ctx.budget_full = True
        return "full"

    staging = tempfile.mkdtemp(prefix="ckpt_", dir=STAGING_DIR)
    try:
        try:
            rel_path, got = stage_file(staging, model, version, fileobj)
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in PERMANENT_HTTP:
                ctx.skiplist.add(fid)
                ctx.skiplist_dirty = True
                print(f"  SKIP (HTTP {status}, won't retry): {model.get('name')} f{fid}")
            else:
                print(f"  FAIL {model.get('name')} f{fid}: {e}", file=sys.stderr)
            return "err"

        # Reflect post-upload state in the files we commit alongside the weight file.
        ctx.archived.add(fid)
        ctx.used_bytes += got
        for name, payload in ctx._state_payload().items():
            with open(os.path.join(staging, name), "wb") as fh:
                fh.write(payload)

        message = (
            f"Archive {model.get('name')} v{version.get('id')} :: {fileobj.get('name')} "
            f"({_gb(got):.1f}GB)\n\n"
            f"https://civitai.com/models/{model.get('id')}\n"
            f"repo now {_tb(ctx.used_bytes):.3f}TB / {STORAGE_LIMIT_TB}TB"
        )
        try:
            upload_folder_with_retry(ctx.api, ARCHIVE_REPO, staging, message)
        except Exception:
            # Upload failed: roll the in-memory tally back so it matches the repo.
            ctx.archived.discard(fid)
            ctx.used_bytes -= got
            raise

        ctx.new_count += 1
        ctx.new_bytes += got
        print(f"  [{ctx.new_count}] archived {rel_path} ({_gb(got):.1f}GB)  "
              f"repo {_tb(ctx.used_bytes):.3f}TB", flush=True)
        return "ok"
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def process_version(ctx, model, version):
    """Archive every wanted weight file of a version. Returns 'cap'/'full' if a
    stop condition was hit, else 'ok'."""
    for fileobj in model_weight_files(version, FILE_TYPES):
        outcome = archive_file(ctx, model, version, fileobj)
        if outcome in ("cap", "full"):
            return outcome
    return "ok"


def run_new(ctx, base_set):
    published_after = datetime.now(timezone.utc) - timedelta(days=SINCE_DAYS)
    print(f"Mode: new (published since {published_after:%Y-%m-%d %H:%M}Z, last {SINCE_DAYS}d)")
    for model, version in iter_lora_versions(base_set, published_after=published_after,
                                             verbose=True, types=MODEL_TYPE):
        if process_version(ctx, model, version) in ("cap", "full"):
            break


def run_heal(ctx, base_set):
    floor = load_backfill_ceiling(ctx.api, ARCHIVE_REPO) or (
        datetime.now(timezone.utc) - timedelta(days=365)
    )
    print(f"Mode: heal (newest -> {floor:%Y-%m-%d %H:%M}Z, filling manifest+skiplist gaps)")
    for model, version in iter_lora_versions(base_set, published_after=floor,
                                             verbose=True, types=MODEL_TYPE):
        if process_version(ctx, model, version) in ("cap", "full"):
            break


def run_backfill(ctx, base_set):
    ceiling = load_backfill_ceiling(ctx.api, ARCHIVE_REPO) or datetime.now(timezone.utc)
    deepest = ceiling
    print(f"Mode: backfill (descending {WINDOW_DAYS}d windows from {ceiling:%Y-%m-%d %H:%M}Z)")
    empty_streak = 0
    stop = False
    try:
        while not stop and empty_streak < MAX_EMPTY_WINDOWS and ceiling > MIN_CEILING:
            floor = ceiling - timedelta(days=WINDOW_DAYS)
            print(f"\n=== Window [{floor:%Y-%m-%d} -> {ceiling:%Y-%m-%d}] ===", flush=True)
            window = []
            for model, version in iter_lora_versions(base_set, published_after=floor,
                                                     verbose=True, types=MODEL_TYPE):
                pub = parse_dt(version.get("publishedAt"))
                if pub is None or pub >= ceiling or pub < floor:
                    continue
                window.append((pub, model, version))
            window.sort(key=lambda t: t[0], reverse=True)
            print(f"  window has {len(window)} candidate versions", flush=True)
            empty_streak = empty_streak + 1 if not window else 0
            for pub, model, version in window:
                if pub < deepest:
                    deepest = pub
                outcome = process_version(ctx, model, version)
                if outcome in ("cap", "full"):
                    stop = True
                    print(f"Stopping ({outcome}); resuming this window next run.")
                    break
            if stop:
                break
            ceiling = floor
            if floor < deepest:
                deepest = floor
            save_backfill_ceiling(ctx.api, ARCHIVE_REPO, deepest)
            print(f"  ceiling persisted at {deepest:%Y-%m-%d %H:%M}Z", flush=True)
    finally:
        final = min(ceiling, deepest)
        save_backfill_ceiling(ctx.api, ARCHIVE_REPO, final)
        print(f"Backfill ceiling saved at {final:%Y-%m-%d %H:%M}Z")


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set", file=sys.stderr)
        return 1

    base_set = set(BASE_MODELS)
    cap_desc = MAX_FILES_PER_RUN if MAX_FILES_PER_RUN > 0 else "unlimited"
    ft_desc = "all" if FILE_TYPES is None else ",".join(FILE_TYPES)
    print(f"Base models: {sorted(base_set)}  |  type: {MODEL_TYPE}  |  files: {ft_desc}")
    print(f"Archive repo: {ARCHIVE_REPO}  |  per-run cap: {cap_desc}  |  "
          f"storage cap: {STORAGE_LIMIT_TB}TB")

    api = HfApi(token=HF_TOKEN)
    try:
        api.create_repo(repo_id=ARCHIVE_REPO, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"create_repo skipped ({e.__class__.__name__}); assuming repo exists", file=sys.stderr)

    archived = set(map(str, load_repo_json(api, ARCHIVE_REPO, MANIFEST_PATH,
                                           {}).get("archived_file_ids", [])))
    skiplist = set(map(str, load_repo_json(api, ARCHIVE_REPO, SKIPLIST_PATH,
                                           {}).get("failed_file_ids", [])))
    used_bytes = int(load_repo_json(api, ARCHIVE_REPO, STORAGE_PATH, {}).get("used_bytes", 0))
    print(f"Manifest: {len(archived)} files  |  skiplist: {len(skiplist)}  |  "
          f"used: {_tb(used_bytes):.3f}TB / {STORAGE_LIMIT_TB}TB")

    ctx = Ctx(api, archived, skiplist, used_bytes)
    if MODE == "backfill":
        run_backfill(ctx, base_set)
    elif MODE == "heal":
        run_heal(ctx, base_set)
    else:
        run_new(ctx, base_set)

    if ctx.skiplist_dirty:
        ctx.save_skiplist_standalone()
        print(f"Skiplist updated: {len(ctx.skiplist)} permanent failures")
    print(f"Done. Archived {ctx.new_count} file(s) / {_gb(ctx.new_bytes):.1f}GB this run; "
          f"{ctx.skipped_size} skipped on size. Repo at {_tb(ctx.used_bytes):.3f}TB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
