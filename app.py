import os
import re
from collections import defaultdict

import gradio as gr
import requests

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

def _repo_used_bytes():
    """Cheap single-call total repo size via HF's usedStorage field."""
    r = requests.get(
        f"https://huggingface.co/api/models/{ARCHIVE_REPO}",
        params={"expand[]": "usedStorage"},
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        timeout=15,
    )
    r.raise_for_status()
    return int(r.json().get("usedStorage") or 0)


def _base_model_of(name):
    """Extract base model from a repo path, handling both old (base/...) and
    new date-prefixed (YYYY-MM-DD/base/...) layouts."""
    parts = name.split("/")
    if parts and (DATE_RE.match(parts[0]) or parts[0] == "undated") and len(parts) > 1:
        return parts[1]
    return parts[0] if parts else "(root)"


def _breakdown():
    """Expensive per-base-model breakdown — only run on demand, never on the timer."""
    api = HfApi(token=HF_TOKEN)
    info = api.repo_info(repo_id=ARCHIVE_REPO, repo_type="model", files_metadata=True)
    per_bm = defaultdict(lambda: [0, 0])
    for s in info.siblings:
        name = s.rfilename
        if name in ("manifest.json", ".gitattributes") or name.endswith(".md"):
            continue
        bm = _base_model_of(name)
        per_bm[bm][0] += 1
        per_bm[bm][1] += s.size or 0
    return [
        [bm, cnt, _fmt_bytes(b)]
        for bm, (cnt, b) in sorted(per_bm.items(), key=lambda kv: -kv[1][1])
    ]


def monitor():
    """Cheap tick: total archived size vs quota. Safe to run every 15s."""
    if not HF_TOKEN:
        return "HF_TOKEN not set."
    try:
        total_bytes = _repo_used_bytes()
    except Exception as e:
        return f"Could not read `{ARCHIVE_REPO}`: {e}"

    pct = (total_bytes / QUOTA_BYTES) * 100 if QUOTA_BYTES else 0
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    return (
        f"## Archive storage\n"
        f"**{_fmt_bytes(total_bytes)}** stored\n\n"
        f"`{bar}` **{pct:.2f}%** of {QUOTA_TB:g} TB quota\n\n"
        f"Repo: https://huggingface.co/{ARCHIVE_REPO}"
    )


def full_refresh():
    """Manual refresh: size + the expensive per-model breakdown."""
    summary = monitor()
    try:
        rows = _breakdown()
    except Exception:
        rows = []
    return summary, rows


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
            "Total size refreshes live every 15s. "
            "Click **Refresh now** for the per-model breakdown (heavier query)."
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
