# Deploying Pelican Press Generator (cheap ZeroGPU setup)

## What's in this repo for the Space
- `app.py` — the Gradio page (content + style upload, Generate button)
- `requirements.txt` / `packages.txt` — Python + system dependencies
- `README.md` — the top block is the Space config

All model weights (stock SDXL, ControlNet-tile, IP-Adapter, BLIP captioner)
download automatically the first time the Space starts. Nothing to source
manually.

## Step 1 — Accounts
1. Create a free Hugging Face account: https://huggingface.co/join
2. Upgrade to **PRO** ($9/mo) to get ZeroGPU: https://huggingface.co/pro

## Step 2 — Create the Space
1. https://huggingface.co/new-space
2. Name: `pelican-press-generator`
3. SDK: **Gradio**
4. Hardware: **ZeroGPU**
5. Visibility: **Private**

## Step 3 — Push this code to the Space
The Space is its own git repo. From this folder:

    git remote add space https://huggingface.co/spaces/<your-username>/pelican-press-generator
    git push space main

Authenticate with a Hugging Face access token (create one at
https://huggingface.co/settings/tokens, "Write" scope).

## Step 4 — First run
The first boot downloads several GB of weights, so it takes a while. After that,
open the Space URL, upload a content image and a style image, and hit Generate.

## If ZeroGPU can't handle it
This pipeline is heavier than typical ZeroGPU apps (a generation takes 1-3 min).
If it times out or runs out of memory, the fallback is a dedicated GPU with
sleep-on-idle:
- Space Settings → Hardware → pick **L40S** or **A100**, enable **sleep after
  inactivity**.
- Roughly $40-130/mo for light use (billed only while awake).
