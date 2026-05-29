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

## Notes & caveats

- Free Spaces sleep when idle and have ephemeral disk — fine for on-demand archiving,
  not for long unattended crawls.
- Respect Civitai creator licenses and HF ToS; some models disallow redistribution.
