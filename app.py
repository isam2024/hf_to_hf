import os
import shutil
import tempfile

import gradio as gr
import requests
from huggingface_hub import HfApi

CIVITAI_API_KEY = os.environ.get("CIVITAI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

CIVITAI_VERSION_API = "https://civitai.com/api/v1/model-versions/{version_id}"
DOWNLOAD_CHUNK = 8 * 1024 * 1024  # 8 MiB


def _civitai_headers():
    h = {"User-Agent": "civitai-to-hf-archiver"}
    if CIVITAI_API_KEY:
        h["Authorization"] = f"Bearer {CIVITAI_API_KEY}"
    return h


def fetch_version_metadata(version_id):
    r = requests.get(
        CIVITAI_VERSION_API.format(version_id=version_id),
        headers=_civitai_headers(),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def pick_primary_file(meta):
    files = meta.get("files", [])
    if not files:
        raise ValueError("This model version has no downloadable files.")
    for f in files:
        if f.get("primary"):
            return f
    return files[0]


def stream_download(url, dest_path, progress, expected_bytes):
    with requests.get(url, headers=_civitai_headers(), stream=True, timeout=60) as r:
        r.raise_for_status()
        downloaded = 0
        with open(dest_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK):
                if not chunk:
                    continue
                fh.write(chunk)
                downloaded += len(chunk)
                if expected_bytes:
                    frac = min(downloaded / expected_bytes, 1.0)
                    progress(frac * 0.6, desc=f"Downloading {downloaded // (1024*1024)} MB")
    return downloaded


def archive(version_id, target_repo, subfolder, progress=gr.Progress()):
    version_id = (version_id or "").strip()
    target_repo = (target_repo or "").strip()
    subfolder = (subfolder or "").strip().strip("/")

    if not version_id:
        return "Please provide a Civitai model-version ID."
    if not target_repo or "/" not in target_repo:
        return "Target repo must look like `username/repo-name`."
    if not HF_TOKEN:
        return "HF_TOKEN secret is not set on this Space."

    try:
        progress(0.0, desc="Fetching Civitai metadata")
        meta = fetch_version_metadata(version_id)
        primary = pick_primary_file(meta)

        filename = primary["name"]
        download_url = primary["downloadUrl"]
        expected_bytes = int(primary.get("sizeKB", 0) * 1024)

        model_name = meta.get("model", {}).get("name", "unknown")
        path_in_repo = f"{subfolder}/{filename}" if subfolder else filename

        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=target_repo, repo_type="model", exist_ok=True)

        tmp_dir = tempfile.mkdtemp(prefix="civitai_")
        local_path = os.path.join(tmp_dir, filename)
        try:
            got = stream_download(download_url, local_path, progress, expected_bytes)
            progress(0.6, desc="Uploading to Hugging Face")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=target_repo,
                repo_type="model",
                commit_message=f"Archive Civitai v{version_id}: {model_name} / {filename}",
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        progress(1.0, desc="Done")
        hf_url = f"https://huggingface.co/{target_repo}/blob/main/{path_in_repo}"
        return (
            f"Archived **{model_name}** ({got // (1024*1024)} MB)\n\n"
            f"File: `{path_in_repo}`\n\n{hf_url}"
        )
    except requests.HTTPError as e:
        return f"HTTP error talking to Civitai/HF: {e}"
    except Exception as e:
        return f"Failed: {e}"


with gr.Blocks(title="Civitai → Hugging Face archiver") as demo:
    gr.Markdown(
        "# Civitai → Hugging Face archiver\n"
        "Paste a Civitai **model-version ID** and a target HF repo. "
        "The file streams Civitai → this Space → the Hub; your machine is never in the path."
    )
    with gr.Row():
        version_in = gr.Textbox(label="Civitai model-version ID", placeholder="e.g. 128713")
        repo_in = gr.Textbox(label="Target HF repo", placeholder="username/civitai-lora-archive")
    subfolder_in = gr.Textbox(label="Subfolder in repo (optional)", placeholder="e.g. style-loras")
    run_btn = gr.Button("Archive", variant="primary")
    out = gr.Markdown()
    run_btn.click(archive, inputs=[version_in, repo_in, subfolder_in], outputs=out)


if __name__ == "__main__":
    demo.launch()
