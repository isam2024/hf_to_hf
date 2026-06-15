---
title: Civitai To HF
emoji: 📦
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Civitai → Hugging Face archiver

A minimal Gradio app that streams a Civitai LoRA/model file straight from Civitai
into a Hugging Face repo. The bytes flow **Civitai → the runner → the Hub** and never
touch your local machine. Designed to run as a free Hugging Face Space.

## How it works

1. You give it a Civitai **model-version ID** and a target HF repo (`username/repo`).
2. It looks up the version via the Civitai API, picks the primary file.
3. It streams the download to the runner's ephemeral disk in 8 MiB chunks, then
   uploads to the Hub and deletes the temp file.

## Secrets / environment variables

Set these as **Space secrets** (Settings → Variables and secrets) — never commit them:

| Name | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face token with **write** access to the target repo. |
| `CIVITAI_API_KEY` | Civitai API key — required for most downloads. |

## Run locally

```bash
pip install -r requirements.txt
export HF_TOKEN=...           # write token
export CIVITAI_API_KEY=...    # civitai key
python app.py
```

## Deploy as a Space

Create a new **Gradio** Space and push this repo to it (or duplicate from GitHub).
Add the two secrets above. The free CPU tier is enough since files only pass through.

## Flux 1 Dev checkpoint crawler

`checkpoint_crawler.py` is the LoRA crawler's big sibling: it archives **Flux 1
Dev checkpoints** (Civitai type `Checkpoint`, base model `Flux.1 D`) to the
secondary HF account (`civitai2026`, ~8.6 TB quota). Unlike LoRAs:

- It grabs **every weight variant** per version (full fp16, bf16, nf4, fp8,
  pruned — all files Civitai tags `type: Model`). Set `FILE_TYPES=Model,UNet`
  or `FILE_TYPES=all` to also pull GGUF UNet / VAE / config files.
- Checkpoints are 6–24 GB, so there is **no batching**: each file is streamed,
  uploaded in its own commit, then deleted — peak disk stays at one file. The
  manifest, storage tally, and skiplist ride along in that same atomic commit,
  so state is resumable at **file granularity** (keyed by Civitai file id).
- A hard `STORAGE_LIMIT_TB` (default 8.5) gates every download; the run stops
  cleanly once the next file would exceed it.

Modes mirror the LoRA crawler (`MODE=new|backfill|heal`); `backfill` descends
the catalog newest-first in `WINDOW_DAYS` windows and persists a ceiling.

Runs unattended via `.github/workflows/checkpoints.yml` (6-hourly cron). It uses
its own secret `HF_TOKEN_CIVITAI2026` (write token for the secondary account)
and reuses `CIVITAI_API_KEY`. The job stages on the runner's `/mnt` volume
(~74 GB) so a single fp16 checkpoint fits. Run it manually from the Actions tab
to pick `new`/`backfill`/`heal`.

```bash
export HF_TOKEN=...                 # civitai2026 write token
export CIVITAI_API_KEY=...
export ARCHIVE_REPO=civitai2026/flux1-dev-checkpoints
MODE=backfill python checkpoint_crawler.py
```

## Notes & caveats

- Free Spaces sleep when idle and have ephemeral disk — fine for on-demand archiving,
  not for long unattended crawls.
- Respect Civitai creator licenses and HF ToS; some models disallow redistribution.
