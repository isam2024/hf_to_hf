import os
import re
import time
from collections import defaultdict

import gradio as gr

from core import HfApi, archive_version, fetch_version_metadata

HF_TOKEN = os.environ.get("HF_TOKEN", "")
ARCHIVE_REPO = os.environ.get("ARCHIVE_REPO", "isam/civitai-lora-archive")
QUOTA_TB = float(os.environ.get("QUOTA_TB", "8.7"))
QUOTA_BYTES = QUOTA_TB * (1024 ** 4)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _fmt_bytes(n):
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


# ---------------- Manual archive tab ----------------

def archive_one(version_id, target_repo, subfolder, progress=gr.Progress()):
    version_id = (version_id or "").strip()
    target_repo = (target_repo or "").strip() or ARCHIVE_REPO
    subfolder = (subfolder or "").strip().strip("/")
    if not version_id:
        return "Please provide a Civitai model-version ID."
    if not HF_TOKEN:
        return "HF_TOKEN secret is not set on this Space."
    try:
        progress(0.0, desc="Fetching Civitai metadata")
        version = fetch_version_metadata(version_id)
        model = version.get("model", {}) or {}
        model.setdefault("id", version.get("modelId", "x"))
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=target_repo, repo_type="model", exist_ok=True)

        def on_prog(frac, done):
            progress(frac * 0.9, desc=f"Downloading {done // (1024*1024)} MB")

        path = archive_version(api, target_repo, model, version, on_progress=on_prog)
        if not path:
            return "This version has no downloadable file."
        progress(1.0, desc="Done")
        return f"Archived `{path}`\n\nhttps://huggingface.co/{target_repo}/blob/main/{path}"
    except Exception as e:
        return f"Failed: {e}"


# ---------------- Monitor tab ----------------

# Cache the (size, count) tuple for a few minutes so the 15s timer
# doesn't saturate the worker by re-listing files every tick.
_SIZE_CACHE = {"at": 0.0, "bytes": 0, "files": 0}
_SIZE_TTL = 300  # seconds


_META = ("manifest.json", "skiplist.json", "backfill_state.json", ".gitattributes")


def _iter_files():
    """Walk the repo tree recursively. list_repo_tree is paginated, so it works
    for arbitrarily large repos (unlike repo_info.siblings which returns 0)."""
    api = HfApi(token=HF_TOKEN)
    for item in api.list_repo_tree(
        repo_id=ARCHIVE_REPO, repo_type="model", recursive=True, expand=True
    ):
        if not hasattr(item, "size"):  # RepoFolder, skip
            continue
        if item.path in _META or item.path.endswith(".md"):
            continue
        sz = item.size
        if getattr(item, "lfs", None):
            sz = item.lfs.size
        yield item.path, sz or 0


def _repo_size_and_count():
    """Sum of all archived file sizes, with TTL cache."""
    now = time.time()
    if now - _SIZE_CACHE["at"] < _SIZE_TTL and _SIZE_CACHE["at"] > 0:
        return _SIZE_CACHE["bytes"], _SIZE_CACHE["files"]
    total = 0
    files = 0
    for _path, sz in _iter_files():
        total += sz
        files += 1
    _SIZE_CACHE.update(at=now, bytes=total, files=files)
    return total, files


def _base_model_of(name):
    """Extract base model from a repo path, handling both old (base/...) and
    new date-prefixed (YYYY-MM-DD/base/...) layouts."""
    parts = name.split("/")
    if parts and (DATE_RE.match(parts[0]) or parts[0] == "undated") and len(parts) > 1:
        return parts[1]
    return parts[0] if parts else "(root)"


def _breakdown():
    """Per-base-model breakdown. Forces a fresh tree walk and updates the size cache."""
    per_bm = defaultdict(lambda: [0, 0])
    total = 0
    files = 0
    for path, sz in _iter_files():
        bm = _base_model_of(path)
        per_bm[bm][0] += 1
        per_bm[bm][1] += sz
        total += sz
        files += 1
    _SIZE_CACHE.update(at=time.time(), bytes=total, files=files)
    return [
        [bm, cnt, _fmt_bytes(b)]
        for bm, (cnt, b) in sorted(per_bm.items(), key=lambda kv: -kv[1][1])
    ]


def monitor():
    """Total archived size vs quota. Cached for 5 min so the 15s tick is cheap."""
    if not HF_TOKEN:
        return "HF_TOKEN not set."
    try:
        total_bytes, file_count = _repo_size_and_count()
    except Exception as e:
        return f"Could not read `{ARCHIVE_REPO}`: {e}"

    pct = (total_bytes / QUOTA_BYTES) * 100 if QUOTA_BYTES else 0
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    return (
        f"## Archive storage\n"
        f"**{_fmt_bytes(total_bytes)}** stored across **{file_count}** files\n\n"
        f"`{bar}` **{pct:.2f}%** of {QUOTA_TB:g} TB quota\n\n"
        f"Repo: https://huggingface.co/{ARCHIVE_REPO}"
    )


def full_refresh():
    """Manual refresh: force a fresh listing (bypassing the cache) and rebuild the breakdown."""
    try:
        rows = _breakdown()
    except Exception:
        rows = []
    return monitor(), rows


with gr.Blocks(title="Civitai → Hugging Face archiver") as demo:
    gr.Markdown("# Civitai → Hugging Face archiver")
    with gr.Tab("Archive a version"):
        gr.Markdown("Stream one Civitai version straight to the Hub. Your machine is never in the path.")
        with gr.Row():
            version_in = gr.Textbox(label="Civitai model-version ID", placeholder="e.g. 128713")
            repo_in = gr.Textbox(label="Target HF repo", value=ARCHIVE_REPO)
        subfolder_in = gr.Textbox(label="Subfolder (optional)")
        run_btn = gr.Button("Archive", variant="primary")
        out = gr.Markdown()
        run_btn.click(archive_one, [version_in, repo_in, subfolder_in], out)

    with gr.Tab("Storage monitor"):
        gr.Markdown(
            "Total size is cached for 5 minutes and refreshes on the 15s tick. "
            "Click **Refresh now** to force a fresh listing and rebuild the per-model breakdown."
        )
        refresh_btn = gr.Button("Refresh now")
        stats_md = gr.Markdown()
        stats_tbl = gr.Dataframe(headers=["Base model", "Files", "Size"], interactive=False)
        # Timer + load: cheap path only (total size), no per-file listing, no State.
        demo.load(monitor, None, stats_md)
        refresh_btn.click(full_refresh, None, [stats_md, stats_tbl])
        try:
            timer = gr.Timer(15)
            timer.tick(monitor, None, stats_md)
        except Exception:
            pass


if __name__ == "__main__":
    demo.launch()
