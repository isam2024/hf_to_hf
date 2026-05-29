"""Auto-archiver: walks Civitai newest-first for the configured base models and
archives any LoRA version not yet in the manifest. Intended to run on a schedule
(GitHub Actions cron).

Files are downloaded to a staging dir and uploaded in BATCHED commits
(one commit per batch via upload_folder) to stay under HF's 128-commits/hour
limit. Streams Civitai -> runner -> HF; nothing persists on the runner."""
import json
import os
import shutil
import sys
import tempfile

from huggingface_hub import HfApi

from core import (
    MANIFEST_PATH,
    iter_lora_versions,
    load_manifest,
    pick_primary_file,
    stage_download,
    upload_folder_with_retry,
)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
ARCHIVE_REPO = os.environ.get("ARCHIVE_REPO", "isam/civitai-lora-archive")
BASE_MODELS = [b.strip() for b in os.environ.get("BASE_MODELS", "Flux.1 D").split(",") if b.strip()]
MAX_FILES_PER_RUN = int(os.environ.get("MAX_FILES_PER_RUN", "0"))  # 0 = unlimited
MAX_FILE_MB = int(os.environ.get("MAX_FILE_MB", "0"))  # 0 = no limit
# Batch thresholds: flush when either is reached. Keeps a batch well under the
# runner's ~14GB disk while minimizing commits (1 commit per batch).
BATCH_FILES = int(os.environ.get("BATCH_FILES", "40"))
BATCH_GB = float(os.environ.get("BATCH_GB", "8"))
BATCH_BYTES = BATCH_GB * (1024 ** 3)


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
    print(f"Manifest: {len(archived)} versions already archived")

    staging = tempfile.mkdtemp(prefix="civitai_batch_")
    new_count = 0
    skipped_size = 0
    batch_files = 0
    batch_bytes = 0
    batch_idx = 0

    def flush():
        nonlocal staging, batch_files, batch_bytes, batch_idx
        if batch_files == 0:
            return
        with open(os.path.join(staging, MANIFEST_PATH), "w") as fh:
            json.dump({"archived_version_ids": sorted(archived, key=lambda x: int(x))}, fh, indent=2)
        batch_idx += 1
        print(f"  -> flushing batch {batch_idx}: {batch_files} files / {batch_bytes/(1024**3):.2f}GB", flush=True)
        upload_folder_with_retry(
            api, ARCHIVE_REPO, staging, f"Archive batch {batch_idx} ({batch_files} files)"
        )
        shutil.rmtree(staging, ignore_errors=True)
        staging = tempfile.mkdtemp(prefix="civitai_batch_")
        batch_files = 0
        batch_bytes = 0

    try:
        for model, version in iter_lora_versions(base_set):
            vid = str(version.get("id"))
            if vid in archived:
                continue
            if MAX_FILES_PER_RUN > 0 and new_count >= MAX_FILES_PER_RUN:
                print(f"Hit per-run cap ({MAX_FILES_PER_RUN}); rest picked up next run.")
                break

            primary = pick_primary_file(version)
            if not primary:
                continue
            size_mb = (primary.get("sizeKB", 0) or 0) / 1024
            if MAX_FILE_MB and size_mb > MAX_FILE_MB:
                print(f"  skip (>{MAX_FILE_MB}MB): {model.get('name')} v{vid} ({size_mb:.0f}MB)")
                skipped_size += 1
                continue

            try:
                result = stage_download(staging, model, version)
            except Exception as e:
                print(f"  FAIL {model.get('name')} v{vid}: {e}", file=sys.stderr)
                continue
            if not result:
                continue
            rel_path, got = result
            archived.add(vid)
            new_count += 1
            batch_files += 1
            batch_bytes += got
            print(f"  [{new_count}] staged {rel_path} ({got/(1024**2):.0f}MB)", flush=True)

            if batch_files >= BATCH_FILES or batch_bytes >= BATCH_BYTES:
                flush()

        flush()  # final partial batch
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"Done. Archived {new_count} new file(s) this run; {skipped_size} skipped on size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
