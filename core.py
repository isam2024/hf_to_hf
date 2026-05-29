"""Shared Civitai -> Hugging Face helpers used by both the Gradio app and the crawler."""
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone

import requests
from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

RETRY_STATUS = {500, 502, 503, 504, 520, 522, 524}


def civitai_get(url, params=None, stream=False, timeout=60, attempts=5):
    """GET against Civitai with retry/backoff on transient errors (5xx, timeouts)."""
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, headers=civitai_headers(), params=params, stream=stream, timeout=timeout)
            if r.status_code in RETRY_STATUS:
                last = requests.HTTPError(f"{r.status_code} from Civitai", response=r)
                r.close()
            else:
                r.raise_for_status()
                return r
        except (requests.ConnectionError, requests.Timeout) as e:
            last = e
        time.sleep(min(2 ** i, 30))
    raise last

CIVITAI_API_KEY = os.environ.get("CIVITAI_API_KEY", "")

MODELS_API = "https://civitai.com/api/v1/models"
VERSION_API = "https://civitai.com/api/v1/model-versions/{version_id}"
DOWNLOAD_CHUNK = 8 * 1024 * 1024  # 8 MiB
MANIFEST_PATH = "manifest.json"
SKIPLIST_PATH = "skiplist.json"
BACKFILL_STATE_PATH = "backfill_state.json"


def civitai_headers():
    h = {"User-Agent": "civitai-to-hf-archiver"}
    if CIVITAI_API_KEY:
        h["Authorization"] = f"Bearer {CIVITAI_API_KEY}"
    return h


def fetch_version_metadata(version_id):
    return civitai_get(VERSION_API.format(version_id=version_id), timeout=30).json()


def pick_primary_file(version):
    files = version.get("files", [])
    if not files:
        return None
    for f in files:
        if f.get("primary"):
            return f
    return files[0]


def parse_dt(s):
    """Parse a Civitai ISO timestamp (or cursor timestamp) to an aware UTC datetime."""
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def iter_lora_versions(base_models, page_limit=100, max_pages=5000, published_after=None, verbose=False):
    """Yield (model, version) pairs for LoRAs matching any of the given base models.

    Pages through Civitai newest-first using cursor pagination. If published_after
    (an aware datetime) is given, only versions published at/after it are yielded,
    and paging stops once the cursor timestamp falls before the cutoff (incremental mode).
    When verbose, prints a heartbeat per page so pure pagination doesn't look frozen.
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
        data = civitai_get(MODELS_API, params=params, timeout=60).json()
        items = data.get("items", [])
        if not items:
            return
        for model in items:
            for version in model.get("modelVersions", []):
                if version.get("baseModel") not in base_models:
                    continue
                if published_after is not None:
                    pub = parse_dt(version.get("publishedAt"))
                    if pub is None or pub < published_after:
                        continue
                yield model, version
        cursor = data.get("metadata", {}).get("nextCursor")
        pages += 1
        if verbose:
            edge = cursor.split("|", 1)[0] if cursor else "end"
            print(f"  ...scanned catalog page {pages} (cursor {edge})", flush=True)
        if not cursor:
            return
        # Incremental early-stop: the cursor's leading timestamp is the sort boundary.
        if published_after is not None:
            cdt = parse_dt(cursor.split("|", 1)[0])
            if cdt is not None and cdt < published_after:
                return


def stream_download(url, dest_path, on_progress=None, expected_bytes=0):
    r = civitai_get(url, stream=True, timeout=120)
    try:
        downloaded = 0
        with open(dest_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK):
                if not chunk:
                    continue
                fh.write(chunk)
                downloaded += len(chunk)
                if on_progress and expected_bytes:
                    on_progress(min(downloaded / expected_bytes, 1.0), downloaded)
    finally:
        r.close()
    return downloaded


def _safe(s):
    return "".join(c if c.isalnum() or c in "-._" else "-" for c in str(s)).strip("-") or "x"


def repo_path_for(model, version, filename):
    bm = _safe(version.get("baseModel", "unknown"))
    mid = model.get("id", "x")
    slug = _safe(model.get("name", "model"))[:60]
    vid = version.get("id", "x")
    pub = parse_dt(version.get("publishedAt"))
    date = pub.strftime("%Y-%m-%d") if pub else "undated"
    return f"{date}/{bm}/{mid}_{slug}/{vid}/{filename}"


def load_manifest(api, repo_id):
    try:
        path = api.hf_hub_download(repo_id=repo_id, filename=MANIFEST_PATH, repo_type="model")
        with open(path) as fh:
            return set(json.load(fh).get("archived_version_ids", []))
    except (EntryNotFoundError, FileNotFoundError):
        return set()


def load_skiplist(api, repo_id):
    """Version IDs that returned permanent download errors (401/403/404) — never retry."""
    try:
        path = api.hf_hub_download(repo_id=repo_id, filename=SKIPLIST_PATH, repo_type="model")
        with open(path) as fh:
            return set(json.load(fh).get("failed_version_ids", []))
    except (EntryNotFoundError, FileNotFoundError):
        return set()


def load_backfill_ceiling(api, repo_id):
    """The upper bound of the next backfill date window (descending). None if uninitialized."""
    try:
        path = api.hf_hub_download(repo_id=repo_id, filename=BACKFILL_STATE_PATH, repo_type="model")
        with open(path) as fh:
            return parse_dt(json.load(fh).get("ceiling"))
    except (EntryNotFoundError, FileNotFoundError):
        return None


def save_backfill_ceiling(api, repo_id, ceiling):
    payload = json.dumps({"ceiling": ceiling.isoformat()}, indent=2).encode()
    api.upload_file(
        path_or_fileobj=payload,
        path_in_repo=BACKFILL_STATE_PATH,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Backfill ceiling -> {ceiling:%Y-%m-%d}",
    )


def save_skiplist(api, repo_id, failed_ids):
    payload = json.dumps(
        {"failed_version_ids": sorted(failed_ids, key=lambda x: int(x))}, indent=2
    ).encode()
    api.upload_file(
        path_or_fileobj=payload,
        path_in_repo=SKIPLIST_PATH,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Update skiplist ({len(failed_ids)} permanent failures)",
    )


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


def _retry_after(resp, default=130):
    try:
        return int(resp.headers.get("Retry-After", default)) + 5
    except (TypeError, ValueError):
        return default


def upload_folder_with_retry(api, repo_id, folder_path, commit_message, attempts=8):
    """Upload a whole folder in ONE commit, backing off on HF 429 commit-rate limits."""
    for i in range(attempts):
        try:
            api.upload_folder(
                folder_path=folder_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )
            return
        except HfHubHTTPError as e:
            resp = getattr(e, "response", None)
            if resp is not None and resp.status_code == 429:
                wait = _retry_after(resp)
                print(f"  HF 429 (commit rate limit); sleeping {wait}s", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"upload_folder failed after {attempts} attempts (rate limited)")


def stage_download(staging_dir, model, version):
    """Download one version's primary file into staging_dir at its repo-relative path.
    Returns (rel_path, size_bytes) or None."""
    primary = pick_primary_file(version)
    if not primary:
        return None
    rel_path = repo_path_for(model, version, primary["name"])
    dest = os.path.join(staging_dir, rel_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        got = stream_download(primary["downloadUrl"], dest, None, int(primary.get("sizeKB", 0) * 1024))
    except Exception:
        if os.path.exists(dest):
            os.remove(dest)
        raise
    return rel_path, got


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
