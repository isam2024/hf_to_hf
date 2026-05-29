import os
from collections import defaultdict

import gradio as gr

from core import HfApi, archive_version, fetch_version_metadata

HF_TOKEN = os.environ.get("HF_TOKEN", "")
ARCHIVE_REPO = os.environ.get("ARCHIVE_REPO", "isam/civitai-lora-archive")
QUOTA_TB = float(os.environ.get("QUOTA_TB", "6"))
QUOTA_BYTES = QUOTA_TB * (1024 ** 4)


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

def storage_stats():
    if not HF_TOKEN:
        return "HF_TOKEN not set.", []
    try:
        api = HfApi(token=HF_TOKEN)
        info = api.repo_info(repo_id=ARCHIVE_REPO, repo_type="model", files_metadata=True)
    except Exception as e:
        return f"Could not read `{ARCHIVE_REPO}`: {e}", []

    per_bm = defaultdict(lambda: [0, 0])  # base model -> [count, bytes]
    total_bytes = 0
    total_files = 0
    for s in info.siblings:
        name = s.rfilename
        if name in ("manifest.json", ".gitattributes") or name.endswith(".md"):
            continue
        size = s.size or 0
        bm = name.split("/", 1)[0] if "/" in name else "(root)"
        per_bm[bm][0] += 1
        per_bm[bm][1] += size
        total_bytes += size
        total_files += 1

    pct = (total_bytes / QUOTA_BYTES) * 100 if QUOTA_BYTES else 0
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))

    recent = ""
    try:
        commits = api.list_repo_commits(repo_id=ARCHIVE_REPO, repo_type="model")[:5]
        lines = [f"- `{c.created_at:%Y-%m-%d %H:%M}` — {c.title[:70]}" for c in commits]
        recent = "\n\n**Recent activity**\n" + "\n".join(lines)
    except Exception:
        pass

    summary = (
        f"## Archive storage\n"
        f"**{_fmt_bytes(total_bytes)}** across **{total_files}** files\n\n"
        f"`{bar}` **{pct:.2f}%** of {QUOTA_TB:.0f} TB quota\n\n"
        f"Repo: https://huggingface.co/{ARCHIVE_REPO}"
        f"{recent}"
    )
    rows = [
        [bm, cnt, _fmt_bytes(b)]
        for bm, (cnt, b) in sorted(per_bm.items(), key=lambda kv: -kv[1][1])
    ]
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
        gr.Markdown("Live view of the archive repo — refreshes every 20s.")
        refresh_btn = gr.Button("Refresh now")
        stats_md = gr.Markdown()
        stats_tbl = gr.Dataframe(headers=["Base model", "Files", "Size"], interactive=False)
        refresh_btn.click(storage_stats, None, [stats_md, stats_tbl])
        demo.load(storage_stats, None, [stats_md, stats_tbl])
        try:
            timer = gr.Timer(20)
            timer.tick(storage_stats, None, [stats_md, stats_tbl])
        except Exception:
            pass


if __name__ == "__main__":
    demo.launch()
