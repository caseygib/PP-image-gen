# Pelican Press Generator
# Gradio wrapper around the InstantStyle-Plus pipeline (infer_style.py), tuned
# for a cheap, lightly-used private Hugging Face Space on ZeroGPU.
#
# Given a CONTENT image and a STYLE image, it generates a new image that applies
# the style while preserving the content's structure.
#
# Differences from the research script, for cost/simplicity:
#   * Stock SDXL base (stabilityai/stable-diffusion-xl-base-1.0) — no custom
#     checkpoint to source.
#   * Lightweight BLIP captioner instead of the ~10GB BLIP2.
#   * CSD style-scoring dropped (it was only used when guidance scales > 0,
#     which we keep at 0) — removes a manual weight download.

# On ZeroGPU, `spaces` must be imported before torch so it can hook CUDA, and
# @spaces.GPU schedules GPU time per call. On a dedicated GPU (or local), there
# is no ZeroGPU scheduler, so we use a no-op decorator and a real CUDA device.
# The ZeroGPU environment sets SPACES_ZERO_GPU.
import os
import sys

# Keep the HF model cache on FAST LOCAL DISK. When a persistent bucket is mounted
# at /data, HF may default the cache there — downloading ~20GB of weights onto
# object storage stalls startup for many minutes. The bucket at /data is only for
# saved batch outputs, not model weights.
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["HF_HUB_CACHE"] = "/tmp/hf_home/hub"
os.makedirs("/tmp/hf_home/hub", exist_ok=True)

# Unbuffered stdout/stderr so startup progress is visible in the Space logs.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

_ZERO_GPU = bool(os.environ.get("SPACES_ZERO_GPU"))
if _ZERO_GPU:
    import spaces
else:
    class _NoSpaces:
        def GPU(self, *args, **kwargs):
            def deco(fn):
                return fn
            return deco

    spaces = _NoSpaces()

import json
import glob
import random
import re
from datetime import datetime

import numpy as np
from PIL import Image

import gradio as gr

# ---------------------------------------------------------------------------
# Work around a known gradio 4.44 bug: gradio_client's schema parser raises
# "TypeError: argument of type 'bool' is not iterable" when a JSON schema
# contains a boolean (e.g. additionalProperties: True/False) while building the
# API info on page load. That crash breaks page rendering AND the launch
# self-check. Make the parser tolerate non-dict schemas.
import gradio_client.utils as _gcu

_orig_get_type = _gcu.get_type
def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)
_gcu.get_type = _safe_get_type

_orig_j2p = _gcu._json_schema_to_python_type
def _safe_j2p(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_j2p(schema, defs)
_gcu._json_schema_to_python_type = _safe_j2p
# ---------------------------------------------------------------------------

import torch
from diffusers import DDIMScheduler, ControlNetModel, AutoencoderKL, StableDiffusionXLPipeline
from transformers import CLIPVisionModelWithProjection
from transformers import BlipProcessor, BlipForConditionalGeneration

from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes
from src.config import RunConfig

from inversion import run as invert
from pipeline_controlnet_sd_xl_img2img import StableDiffusionXLControlNetImg2ImgPipeline

# ---------------------------------------------------------------------------
# Model ids / paths
# ---------------------------------------------------------------------------
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"   # stock SDXL, auto-downloads
CONTROLNET = "xinsir/controlnet-tile-sdxl-1.0"            # auto-downloads
IP_ADAPTER = "h94/IP-Adapter"                             # auto-downloads
CAPTIONER  = "Salesforce/blip-image-captioning-large"    # ~0.9GB, auto-downloads

# On ZeroGPU CUDA isn't physically attached at import, but `.to("cuda")` is
# intercepted by the `spaces` shim, so we still target cuda.
DEVICE = "cuda" if (_ZERO_GPU or torch.cuda.is_available()) else "cpu"
# bfloat16 (not float16): same memory as fp16 but fp32's numeric range, which
# avoids the NaN overflow that fp16 hits during ReNoise inversion.
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def resize_img(input_image, max_side=1280, min_side=1024,
               mode=Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    ratio = min_side / min(h, w)
    w, h = round(ratio * w), round(ratio * h)
    ratio = max_side / max(h, w)
    input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
    w_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
    h_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    return input_image.resize([w_new, h_new], mode)


# ---------------------------------------------------------------------------
# Load models once at startup.
# ---------------------------------------------------------------------------
print("Loading BLIP captioner ...")
caption_processor = BlipProcessor.from_pretrained(CAPTIONER)
caption_model = BlipForConditionalGeneration.from_pretrained(
    CAPTIONER, torch_dtype=DTYPE
).to(DEVICE)
caption_model.eval()

print("Loading fp16-safe SDXL VAE ...")
# Stock SDXL's VAE overflows in fp16 and decodes to a black image; this
# drop-in replacement is numerically stable in fp16.
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=DTYPE
).to(DEVICE)

print("Loading ControlNet (tile) ...")
controlnet = ControlNetModel.from_pretrained(
    CONTROLNET, torch_dtype=DTYPE, use_safetensors=True
).to(DEVICE)

print("Loading IP-Adapter image encoder ...")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    IP_ADAPTER, subfolder="models/image_encoder", torch_dtype=DTYPE
).to(DEVICE)

print("Loading main ControlNet img2img pipeline ...")
pipe_inference = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    vae=vae,                   # fp16-safe VAE (avoids black-image decode)
    clip_model=None,           # CSD scoring disabled
    image_encoder=image_encoder,
    torch_dtype=DTYPE,
    use_safetensors=True,
    variant="fp16",
).to(DEVICE)
pipe_inference.scheduler = DDIMScheduler.from_config(pipe_inference.scheduler.config)
pipe_inference.unet.enable_gradient_checkpointing()
pipe_inference.load_ip_adapter(
    [IP_ADAPTER, IP_ADAPTER],
    subfolder=["sdxl_models", "sdxl_models"],
    weight_name=["ip-adapter_sdxl_vit-h.safetensors", "ip-adapter_sdxl_vit-h.safetensors"],
    image_encoder_folder=None,
)

print("Startup complete (text-to-image pipeline loads lazily on first use).", flush=True)


# --- Text-to-image (InstantStyle) pipeline: loaded lazily on first use -------
# Lighter path — no inversion/ControlNet. Loading it on demand (instead of at
# startup) keeps cold-start fast; the first description/batch generation pays a
# one-time load cost, then it's cached.
_pipe_t2i = None


def get_t2i():
    global _pipe_t2i
    if _pipe_t2i is None:
        print("Loading text-to-image (InstantStyle) pipeline ...", flush=True)
        p = StableDiffusionXLPipeline.from_pretrained(
            BASE_MODEL,
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=DTYPE,
            use_safetensors=True,
            variant="fp16",
            add_watermarker=False,
        ).to(DEVICE)
        p.load_ip_adapter(
            IP_ADAPTER,
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl_vit-h.safetensors",
            image_encoder_folder=None,
        )
        _pipe_t2i = p
    return _pipe_t2i


def generate_caption(image: Image.Image) -> str:
    inputs = caption_processor(images=image, return_tensors="pt").to(DEVICE, DTYPE)
    generated_ids = caption_model.generate(**inputs, max_length=50, num_beams=5)
    return caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# Per-request generation (runs on the ZeroGPU-allocated A100).
# ---------------------------------------------------------------------------
@spaces.GPU(duration=90)
def stylize(content_image, style_image, steps, seed,
            style_strength, structure_strength,
            progress=gr.Progress()):
    if content_image is None or style_image is None:
        raise gr.Error("Please provide both a content image and a style image.")

    content_image = resize_img(content_image.convert("RGB"))
    style_image = style_image.convert("RGB")

    progress(0.1, desc="Reading content image")
    prompt = generate_caption(content_image)

    # --- inversion: recover the content latent ---
    progress(0.3, desc="Analyzing structure")
    config = RunConfig(
        model_type=Model_Type.SDXL,
        num_inference_steps=int(steps),
        num_inversion_steps=int(steps),
        num_renoise_steps=1,
        scheduler_type=Scheduler_Type.DDIM,
        perform_noise_correction=False,
        seed=int(seed),
    )
    pipe_inversion, pipe_inv_infer = get_pipes(
        Model_Type.SDXL, Scheduler_Type.DDIM, device=DEVICE, model_name=BASE_MODEL
    )
    # Use the fp16-safe VAE for inversion encoding too — stock SDXL's VAE
    # overflows in fp16 and yields NaN latents (which decode to black).
    pipe_inversion.vae = vae
    pipe_inv_infer.vae = vae
    _, inv_latent, _, _ = invert(
        content_image, prompt, config,
        pipe_inversion=pipe_inversion, pipe_inference=pipe_inv_infer,
        do_reconstruction=False,
    )
    print(f"[diag] inv_latent nan={torch.isnan(inv_latent).any().item()} "
          f"min={inv_latent.float().min().item():.3f} "
          f"max={inv_latent.float().max().item():.3f}", flush=True)
    del pipe_inversion, pipe_inv_infer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # style vs structure trade-offs (sliders)
    pipe_inference.set_ip_adapter_scale(
        [0.2, {"up": {"block_0": [0.0, float(style_strength), 0.0]}}]
    )

    progress(0.55, desc="Applying style")
    images = pipe_inference(
        prompt=prompt,
        negative_prompt="lowres, low quality, worst quality, deformed, noisy, blurry",
        ip_adapter_image=[content_image, style_image],
        guidance_scale=5,
        num_inference_steps=int(steps),
        image=inv_latent,
        control_image=resize_img(content_image),
        controlnet_conditioning_scale=float(structure_strength),
        denoising_start=0.0001,
        style_embeddings_clip=None,
        content_embeddings_clip=None,
        style_guidance_scale=0,
        content_guidance_scale=0,
    ).images

    _arr = np.array(images[0])
    print(f"[diag] output shape={_arr.shape} min={_arr.min()} "
          f"max={_arr.max()} mean={_arr.mean():.2f}", flush=True)

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    progress(1.0, desc="Done")
    return images[0]


# ---------------------------------------------------------------------------
# Per-request generation: text prompt + style image (InstantStyle text2img)
# ---------------------------------------------------------------------------
@spaces.GPU(duration=90)
def stylize_from_text(style_image, prompt, steps, seed, style_strength,
                      progress=gr.Progress()):
    if style_image is None:
        raise gr.Error("Please provide a style image.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please describe what you want to generate.")

    style_image = style_image.convert("RGB")
    progress(0.2, desc="Generating")
    image = _t2i(style_image, prompt, seed, steps, style_strength)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    progress(1.0, desc="Done")
    return image


# ---------------------------------------------------------------------------
# Shared text-to-image core (used by the description tab and the batch tab)
# ---------------------------------------------------------------------------
def _t2i(style_image, prompt, seed, steps, style_strength):
    pipe_t2i = get_t2i()
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    # Inject style only into the style-relevant IP-Adapter block so content
    # comes from the text prompt, not the style image.
    pipe_t2i.set_ip_adapter_scale({"up": {"block_0": [0.0, float(style_strength), 0.0]}})
    return pipe_t2i(
        prompt=prompt,
        negative_prompt="lowres, low quality, worst quality, deformed, noisy, blurry",
        ip_adapter_image=style_image,
        guidance_scale=7.0,
        num_inference_steps=int(steps),
        generator=generator,
    ).images[0]


# ---------------------------------------------------------------------------
# Batch generation: N breeds x M poses, cohesive per breed, persisted to disk.
# Results are written under DATA_ROOT so they survive restarts when the Space
# has persistent storage mounted at /data.
# ---------------------------------------------------------------------------
DATA_ROOT = "/data" if os.path.isdir("/data") else os.path.join(os.getcwd(), "batch_data")
BATCH_ROOT = os.path.join(DATA_ROOT, "batches")
os.makedirs(BATCH_ROOT, exist_ok=True)


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.strip().lower()).strip("-")[:60] or "x"


def _manifest_path(batch_id):
    return os.path.join(BATCH_ROOT, batch_id, "manifest.json")


def _load_manifest(batch_id):
    with open(_manifest_path(batch_id)) as f:
        return json.load(f)


def _save_manifest(m):
    with open(_manifest_path(m["batch_id"]), "w") as f:
        json.dump(m, f, indent=2)


def _gallery_from_manifest(m):
    items = []
    bdir = os.path.join(BATCH_ROOT, m["batch_id"])
    for breed in m["breeds"]:
        for pose in m["poses"]:
            cell = m["cells"].get(f"{breed}||{pose}")
            if cell and os.path.exists(os.path.join(bdir, cell["file"])):
                items.append((os.path.join(bdir, cell["file"]), f"{breed} — {pose}"))
    return items


def _cell_keys(m):
    keys = []
    for breed in m["breeds"]:
        for pose in m["poses"]:
            if m["cells"].get(f"{breed}||{pose}"):
                keys.append(f"{breed}||{pose}")
    return keys


def list_batches():
    if not os.path.isdir(BATCH_ROOT):
        return []
    ids = [d for d in os.listdir(BATCH_ROOT) if os.path.isfile(_manifest_path(d))]
    return sorted(ids, reverse=True)


def load_latest_batch():
    ids = list_batches()
    if not ids:
        return [], None, gr.update(choices=[], value=None)
    m = _load_manifest(ids[0])
    return _gallery_from_manifest(m), ids[0], gr.update(choices=ids, value=ids[0])


def switch_batch(batch_id):
    if not batch_id:
        return [], None, ""
    m = _load_manifest(batch_id)
    return (_gallery_from_manifest(m), batch_id,
            f"Loaded batch {batch_id} — {len(m['cells'])} images.")


def refresh_picker():
    return gr.update(choices=list_batches())


def run_batch(style_image, template, poses_text, breeds_text, steps, style_strength,
              progress=gr.Progress()):
    if style_image is None:
        raise gr.Error("Please provide a style reference image.")
    if "{breed}" not in (template or ""):
        raise gr.Error("The prompt template must contain {breed}.")
    poses = [p.strip() for p in poses_text.splitlines() if p.strip()]
    breeds = [b.strip() for b in breeds_text.splitlines() if b.strip()]
    if not poses or not breeds:
        raise gr.Error("Enter at least one pose and one breed (one per line).")

    style_image = style_image.convert("RGB")
    batch_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    bdir = os.path.join(BATCH_ROOT, batch_id)
    os.makedirs(bdir, exist_ok=True)
    style_image.save(os.path.join(bdir, "style.png"))

    manifest = {
        "batch_id": batch_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "prompt_template": template,
        "poses": poses,
        "breeds": breeds,
        "steps": int(steps),
        "style_strength": float(style_strength),
        "cells": {},
    }
    _save_manifest(manifest)

    total = len(breeds) * len(poses)
    done = 0
    for bi, breed in enumerate(breeds):
        # Same seed for all poses of a breed -> cohesive "same dog" feel.
        breed_seed = 100000 + bi
        for pose in poses:
            prompt = f"{template.format(breed=breed)}, {pose}"
            img = _t2i(style_image, prompt, breed_seed, steps, style_strength)
            fname = f"{_slug(breed)}__{_slug(pose)}.png"
            img.save(os.path.join(bdir, fname))
            manifest["cells"][f"{breed}||{pose}"] = {
                "seed": breed_seed, "file": fname, "prompt": prompt,
            }
            done += 1
        _save_manifest(manifest)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        progress(done / total, desc=f"{breed} done ({done}/{total})")
        yield (_gallery_from_manifest(manifest), batch_id,
               f"Generated {done}/{total}. Batch: {batch_id}")

    yield (_gallery_from_manifest(manifest), batch_id,
           f"✅ Done — {done} images. Batch: {batch_id}")


def on_gallery_select(batch_id, evt: gr.SelectData):
    if not batch_id:
        return None, ""
    m = _load_manifest(batch_id)
    keys = _cell_keys(m)
    if evt.index is None or not (0 <= evt.index < len(keys)):
        return None, ""
    key = keys[evt.index]
    breed, pose = key.split("||", 1)
    return key, f"Selected: {breed} — {pose}"


def regenerate_cell(batch_id, sel_key):
    if not batch_id or not sel_key:
        raise gr.Error("Click an image in the gallery to select it first.")
    m = _load_manifest(batch_id)
    bdir = os.path.join(BATCH_ROOT, batch_id)
    breed, pose = sel_key.split("||", 1)
    style_image = Image.open(os.path.join(bdir, "style.png")).convert("RGB")
    new_seed = random.randint(0, 2 ** 31 - 1)
    prompt = f"{m['prompt_template'].format(breed=breed)}, {pose}"
    img = _t2i(style_image, prompt, new_seed, m["steps"], m["style_strength"])
    fname = f"{_slug(breed)}__{_slug(pose)}.png"
    img.save(os.path.join(bdir, fname))
    m["cells"][sel_key] = {"seed": new_seed, "file": fname, "prompt": prompt}
    _save_manifest(m)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return _gallery_from_manifest(m), f"Regenerated: {breed} — {pose} (seed {new_seed})"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Pelican Press Generator") as demo:
    gr.Markdown("# 🕊️ Pelican Press Generator")

    with gr.Tabs():
        # ---- Tab 1: text description + style image ----
        with gr.Tab("From a description"):
            gr.Markdown(
                "Upload a **style** image and **describe** what you want. "
                "The generator creates a new image of your description in that style."
            )
            with gr.Row():
                t2i_style = gr.Image(label="Style image", type="pil", height=360)
                t2i_prompt = gr.Textbox(
                    label="Describe what you want",
                    placeholder="e.g. a golden retriever sitting down",
                    lines=4,
                )
            with gr.Accordion("Advanced settings", open=False):
                t2i_steps = gr.Slider(10, 50, value=30, step=1,
                                      label="Quality steps (higher = better, slower)")
                t2i_seed = gr.Number(value=7865, label="Seed", precision=0)
                t2i_style_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.1,
                                               label="Style strength")
            t2i_btn = gr.Button("Generate", variant="primary")
            t2i_output = gr.Image(label="Result", height=480, format="png")
            t2i_btn.click(
                stylize_from_text,
                inputs=[t2i_style, t2i_prompt, t2i_steps, t2i_seed, t2i_style_strength],
                outputs=t2i_output,
            )

        # ---- Tab 2: content image + style image (original) ----
        with gr.Tab("From an image"):
            gr.Markdown(
                "Upload a **content** image and a **style** image. The generator "
                "applies the style while preserving the content's structure."
            )
            with gr.Row():
                content_in = gr.Image(label="Content image", type="pil", height=360)
                style_in = gr.Image(label="Style image", type="pil", height=360)
            with gr.Accordion("Advanced settings", open=False):
                steps = gr.Slider(10, 50, value=20, step=1,
                                  label="Quality steps (higher = better, slower)")
                seed = gr.Number(value=7865, label="Seed", precision=0)
                style_strength = gr.Slider(0.0, 2.0, value=1.2, step=0.1, label="Style strength")
                structure_strength = gr.Slider(0.0, 1.0, value=0.4, step=0.05,
                                               label="Structure preservation")
            run_btn = gr.Button("Generate", variant="primary")
            output = gr.Image(label="Result", height=480, format="png")

            run_btn.click(
                stylize,
                inputs=[content_in, style_in, steps, seed, style_strength, structure_strength],
                outputs=output,
            )

        # ---- Tab 3: batch (breeds x poses) ----
        with gr.Tab("Batch (breeds × poses)"):
            gr.Markdown(
                "Generate a cohesive set of poses for many subjects at once. "
                "Each subject uses one style image and a fixed seed, so its poses "
                "look like the same subject. Results are saved and can be reloaded "
                "and regenerated later. **Runs on the dedicated GPU; a full 40-item "
                "batch takes ~20–35 min.**"
            )
            with gr.Row():
                batch_style = gr.Image(label="Style reference image", type="pil", height=300)
                with gr.Column():
                    batch_template = gr.Textbox(
                        label="Prompt template (must include {breed})",
                        value="a {breed} antique sketch",
                    )
                    batch_poses = gr.Textbox(
                        label="Poses (one per line)",
                        value="sitting down\nstanding\nportrait",
                        lines=4,
                    )
                    batch_breeds = gr.Textbox(
                        label="Subjects / breeds (one per line)",
                        value="labrador retriever\ngolden retriever\ngerman shepherd",
                        lines=8,
                    )
            with gr.Accordion("Advanced settings", open=False):
                batch_steps = gr.Slider(10, 50, value=30, step=1,
                                        label="Quality steps (higher = better, slower)")
                batch_style_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.1,
                                                 label="Style strength")
            batch_run = gr.Button("Run batch", variant="primary")
            batch_status = gr.Markdown("")

            batch_state = gr.State(None)   # current batch_id
            sel_state = gr.State(None)     # selected cell key

            batch_picker = gr.Dropdown(label="Load a past batch", choices=[],
                                       interactive=True)
            batch_gallery = gr.Gallery(label="Results (click an image to select it)",
                                       columns=3, height=650, object_fit="contain",
                                       show_label=True)
            regen_btn = gr.Button("Regenerate selected image")

            batch_run.click(
                run_batch,
                inputs=[batch_style, batch_template, batch_poses, batch_breeds,
                        batch_steps, batch_style_strength],
                outputs=[batch_gallery, batch_state, batch_status],
            ).then(refresh_picker, None, batch_picker)

            batch_gallery.select(on_gallery_select, inputs=[batch_state],
                                 outputs=[sel_state, batch_status])
            regen_btn.click(regenerate_cell, inputs=[batch_state, sel_state],
                            outputs=[batch_gallery, batch_status])
            batch_picker.change(switch_batch, inputs=[batch_picker],
                                outputs=[batch_gallery, batch_state, batch_status])

    # On page load, restore the most recent batch (survives restarts when
    # persistent storage is mounted).
    demo.load(load_latest_batch, None, [batch_gallery, batch_state, batch_picker])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=[DATA_ROOT])
