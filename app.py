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

# --- Text-to-image (InstantStyle) pipeline: text prompt + style image -------
# Lighter path — no inversion/ControlNet. Style is injected only into the
# style-relevant IP-Adapter block, so content comes from the text prompt.
print("Loading text-to-image (InstantStyle) pipeline ...")
pipe_t2i = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL,
    vae=vae,
    image_encoder=image_encoder,
    torch_dtype=DTYPE,
    use_safetensors=True,
    variant="fp16",
    add_watermarker=False,
).to(DEVICE)
pipe_t2i.load_ip_adapter(
    IP_ADAPTER,
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl_vit-h.safetensors",
    image_encoder_folder=None,
)
print("Startup complete.")


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
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    # Inject style only into the style-relevant IP-Adapter block so the content
    # is driven by the text prompt, not the style image.
    pipe_t2i.set_ip_adapter_scale({"up": {"block_0": [0.0, float(style_strength), 0.0]}})

    progress(0.2, desc="Generating")
    image = pipe_t2i(
        prompt=prompt,
        negative_prompt="lowres, low quality, worst quality, deformed, noisy, blurry",
        ip_adapter_image=style_image,
        guidance_scale=7.0,
        num_inference_steps=int(steps),
        generator=generator,
    ).images[0]

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    progress(1.0, desc="Done")
    return image


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

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
