"""Auto-archiver: walks Civitai newest-first for the configured base models and
archives any LoRA version not yet in the manifest. Intended to run on a schedule
(GitHub Actions cron). Streams Civitai -> runner -> HF; nothing persists on the runner."""
import os
import sys

from huggingface_hub import HfApi

from core import archive_version, iter_lora_versions, load_manifest, save_manifest

HF_TOKEN = os.environ.get("HF_TOKEN", "")
ARCHIVE_REPO = os.environ.get("ARCHIVE_REPO", "isam/civitai-lora-archive")
# Comma-separated Civitai base-model strings. Confirm exact spellings against Civitai.
BASE_MODELS = [b.strip() for b in os.environ.get("BASE_MODELS", "Flux.1 D").split(",") if b.strip()]
MAX_FILES_PER_RUN = int(os.environ.get("MAX_FILES_PER_RUN", "100"))
MAX_FILE_MB = int(os.environ.get("MAX_FILE_MB", "0"))  # 0 = no limit
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "25"))  # manifest commit cadence


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set", file=sys.stderr)
        return 1

    base_set = set(BASE_MODELS)
    print(f"Base models: {sorted(base_set)}")
    print(f"Archive repo: {ARCHIVE_REPO}  |  per-run cap: {MAX_FILES_PER_RUN}")

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=ARCHIVE_REPO, repo_type="model", exist_ok=True)
    archived = load_manifest(api, ARCHIVE_REPO)
    print(f"Manifest: {len(archived)} versions already archived")

    new_count = 0
    skipped_size = 0
    for model, version in iter_lora_versions(base_set):
        vid = str(version.get("id"))
        if vid in archived:
            continue
        if new_count >= MAX_FILES_PER_RUN:
            print(f"Hit per-run cap ({MAX_FILES_PER_RUN}); remaining will be picked up next run.")
            break

        from core import pick_primary_file

        primary = pick_primary_file(version)
        if not primary:
            continue
        size_mb = (primary.get("sizeKB", 0) or 0) / 1024
        if MAX_FILE_MB and size_mb > MAX_FILE_MB:
            print(f"  skip (>{MAX_FILE_MB}MB): {model.get('name')} v{vid} ({size_mb:.0f}MB)")
            skipped_size += 1
            continue

        try:
            path = archive_version(api, ARCHIVE_REPO, model, version)
            if path:
                archived.add(vid)
                new_count += 1
                print(f"  [{new_count}] {path} ({size_mb:.0f}MB)", flush=True)
                if new_count % SAVE_EVERY == 0:
                    save_manifest(api, ARCHIVE_REPO, archived)
        except Exception as e:
            print(f"  FAIL {model.get('name')} v{vid}: {e}", file=sys.stderr)

    if new_count:
        save_manifest(api, ARCHIVE_REPO, archived)
    print(f"Done. Archived {new_count} new file(s) this run; {skipped_size} skipped on size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
