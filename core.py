"""Shared Civitai -> Hugging Face helpers used by both the Gradio app and the crawler."""
import json
import os
import shutil
import tempfile

import requests
from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError

CIVITAI_API_KEY = os.environ.get("CIVITAI_API_KEY", "")

MODELS_API = "https://civitai.com/api/v1/models"
VERSION_API = "https://civitai.com/api/v1/model-versions/{version_id}"
DOWNLOAD_CHUNK = 8 * 1024 * 1024  # 8 MiB
MANIFEST_PATH = "manifest.json"


def civitai_headers():
    h = {"User-Agent": "civitai-to-hf-archiver"}
    if CIVITAI_API_KEY:
        h["Authorization"] = f"Bearer {CIVITAI_API_KEY}"
    return h


def fetch_version_metadata(version_id):
    r = requests.get(
        VERSION_API.format(version_id=version_id), headers=civitai_headers(), timeout=30
    )
    r.raise_for_status()
    return r.json()


def pick_primary_file(version):
    files = version.get("files", [])
    if not files:
        return None
    for f in files:
        if f.get("primary"):
            return f
    return files[0]


def iter_lora_versions(base_models, page_limit=100, max_pages=50):
    """Yield (model, version) pairs for LoRAs matching any of the given base models.

    Pages through Civitai newest-first using cursor pagination.
    """
    cursor = None
    pages = 0
    while pages < max_pages:
        params = {
            "types": "LORA",
            "sort": "Newest",
            "limit": page_limit,
            "baseModels": list(base_models),
        }
        if cursor:
            params["cursor"] = cursor
        r = requests.get(MODELS_API, headers=civitai_headers(), params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", [])
        if not items:
            return
        for model in items:
            for version in model.get("modelVersions", []):
                if version.get("baseModel") in base_models:
                    yield model, version
        cursor = data.get("metadata", {}).get("nextCursor")
        pages += 1
        if not cursor:
            return


def stream_download(url, dest_path, on_progress=None, expected_bytes=0):
    with requests.get(url, headers=civitai_headers(), stream=True, timeout=120) as r:
        r.raise_for_status()
        downloaded = 0
        with open(dest_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK):
                if not chunk:
                    continue
                fh.write(chunk)
                downloaded += len(chunk)
                if on_progress and expected_bytes:
                    on_progress(min(downloaded / expected_bytes, 1.0), downloaded)
    return downloaded


def _safe(s):
    return "".join(c if c.isalnum() or c in "-._" else "-" for c in str(s)).strip("-") or "x"


def repo_path_for(model, version, filename):
    bm = _safe(version.get("baseModel", "unknown"))
    mid = model.get("id", "x")
    slug = _safe(model.get("name", "model"))[:60]
    vid = version.get("id", "x")
    return f"{bm}/{mid}_{slug}/{vid}/{filename}"


def load_manifest(api, repo_id):
    try:
        path = api.hf_hub_download(repo_id=repo_id, filename=MANIFEST_PATH, repo_type="model")
        with open(path) as fh:
            return set(json.load(fh).get("archived_version_ids", []))
    except (EntryNotFoundError, FileNotFoundError):
        return set()


def save_manifest(api, repo_id, archived_ids):
    payload = json.dumps(
        {"archived_version_ids": sorted(archived_ids, key=lambda x: int(x))}, indent=2
    ).encode()
    api.upload_file(
        path_or_fileobj=payload,
        path_in_repo=MANIFEST_PATH,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Update manifest ({len(archived_ids)} versions)",
    )


def archive_version(api, repo_id, model, version, on_progress=None):
    """Download one version's primary file and upload it. Returns the repo path or None."""
    primary = pick_primary_file(version)
    if not primary:
        return None
    filename = primary["name"]
    path_in_repo = repo_path_for(model, version, filename)
    tmp_dir = tempfile.mkdtemp(prefix="civitai_")
    local_path = os.path.join(tmp_dir, filename)
    try:
        stream_download(
            primary["downloadUrl"], local_path, on_progress, int(primary.get("sizeKB", 0) * 1024)
        )
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Archive {model.get('name')} v{version.get('id')}",
        )
        return path_in_repo
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
