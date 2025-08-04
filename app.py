#!/usr/bin/env python
"""
Realtime knee‑X‑ray demo with TWO models
────────────────────────────────────────
• Base  : google/medgemma‑4b‑it
• Fine‑tuned: LoRA adapter you provide
Both answers are returned for every query.

Python ≥3.8 · Gradio ≥3.1 (no 4‑only calls)

Example
-------
python app_medgemma_dual_demo.py \
       --adapter_dir medgemma-4b-it-sft-lora-knee-ddp \
       --base_model_id google/medgemma-4b-it \
       --bf16 --share
"""

import functools, argparse
from typing import Tuple, Optional

import gradio as gr
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel, PeftConfig

# ─────────────────────── Single‑model loader helpers ──────────────────────
def _select_dtype(bf16: bool, cpu: bool):
    return torch.bfloat16 if (bf16 and not cpu) else torch.float32

@functools.lru_cache(maxsize=1)
def load_base_model(base_id: str, bf16: bool, cpu: bool):
    dtype   = _select_dtype(bf16, cpu)
    device  = torch.device("cpu" if cpu else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model   = AutoModelForImageTextToText.from_pretrained(base_id, torch_dtype=dtype).to(device).eval()
    proc    = AutoProcessor.from_pretrained(base_id)
    proc.tokenizer.model_max_length = 1024; proc.tokenizer.padding_side = "right"
    eos_id  = proc.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    return model, proc, eos_id, proc.tokenizer.pad_token_id, device

@functools.lru_cache(maxsize=1)
def load_lora_model(base_id: str, adapter_dir: str, bf16: bool, cpu: bool):
    dtype   = _select_dtype(bf16, cpu)
    device  = torch.device("cpu" if cpu else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    base    = AutoModelForImageTextToText.from_pretrained(base_id, torch_dtype=dtype)
    model   = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    model.to(device).eval()

    proc    = AutoProcessor.from_pretrained(base_id)
    proc.tokenizer.model_max_length = 1024; proc.tokenizer.padding_side = "right"
    eos_id  = proc.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    return model, proc, eos_id, proc.tokenizer.pad_token_id, device

# ───────────────────────── Mosaic helper ───────────────────────────────────
def build_mosaic(ap=None, pa=None, sky=None, lat1=None, lat2=None, size=896):
    canvas, h2 = Image.new("L", (size, size)), size // 2
    norm = lambda im: im.convert("L").resize((h2, h2), Image.LANCZOS)
    if ap  : canvas.paste(norm(ap),   (0, 0))
    if pa  : canvas.paste(norm(pa),   (h2, 0))
    if sky : canvas.paste(norm(sky),  (0, h2))
    lat_blk = Image.new("L", (h2, h2)); q = h2 // 2
    if lat1: lat_blk.paste(lat1.convert("L").resize((q, h2), Image.LANCZOS),(0, 0))
    if lat2: lat_blk.paste(lat2.convert("L").resize((q, h2), Image.LANCZOS),(q, 0))
    canvas.paste(lat_blk, (h2, h2))
    return canvas

# ───────────────────────── Inference core ──────────────────────────────────
def _infer_one(model, proc, eos_id, pad_id, device, img_rgb, user_prompt) -> str:
    chat = [{"role":"user","content":[
        {"type":"text","text":"<task=report>"},
        {"type":"image"},
        {"type":"text","text":user_prompt}
    ]}]
    prompt = proc.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=[prompt], images=[[img_rgb]], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                             no_repeat_ngram_size=4, repetition_penalty=1.1,
                             eos_token_id=eos_id, pad_token_id=pad_id)
    return proc.tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip()

def generate(
    mode, single_img, ap, pa, sky, lat1, lat2,
    prompt_choice, custom_prompt,
    base_model_id, adapter_dir, bf16, cpu
) -> Tuple[str, str, Image.Image]:

    # ---------------- build image -----------------
    if mode == "Single image":
        if single_img is None:
            return "⚠️ Upload an image.", "", None
        img_rgb = single_img.convert("RGB")
    else:
        if all(v is None for v in (ap, pa, sky, lat1, lat2)):
            return "⚠️ Provide at least one knee view.", "", None
        img_rgb = build_mosaic(ap, pa, sky, lat1, lat2).convert("RGB")

    # ---------------- prompt ----------------------
    if prompt_choice == "Custom":
        user_prompt = custom_prompt.strip()
        if not user_prompt:
            return "⚠️ Custom prompt is empty.", "", img_rgb
    elif prompt_choice.startswith("Describe"):
        user_prompt = "Describe this X‑ray image."
    else:
        user_prompt = "Generate the report (impression) of the following X‑ray images."

    # ---------------- inference (Base) ------------
    base_m, base_p, base_eos, base_pad, base_dev = load_base_model(base_model_id, bf16, cpu)
    answer_base = _infer_one(base_m, base_p, base_eos, base_pad, base_dev, img_rgb, user_prompt)

    # ---------------- inference (LoRA) ------------
    lora_m, lora_p, lora_eos, lora_pad, lora_dev = load_lora_model(
        base_model_id, adapter_dir, bf16, cpu)
    answer_lora = _infer_one(lora_m, lora_p, lora_eos, lora_pad, lora_dev, img_rgb, user_prompt)

    return f"**Base MedGemma**\n\n{answer_base}", f"**Fine‑tuned (LoRA)**\n\n{answer_lora}", img_rgb

# ───────────────────────── UI builder ──────────────────────────────────────
def build_demo(base_model_id, adapter_dir, bf16, cpu):
    with gr.Blocks(title="MedGemma – Base vs. LoRA") as demo:
        gr.Markdown("## 🦵 Knee X‑ray reasoning – compare *base* vs *fine‑tuned*")

        # hidden states
        st_base = gr.State(base_model_id); st_ad = gr.State(adapter_dir)
        st_bf16 = gr.State(bf16); st_cpu = gr.State(cpu)

        mode = gr.Radio(["Single image", "Multi‑view"], label="Input mode", value="Single image")

        single_up = gr.Image(type="pil", label="Upload image")

        with gr.Row(visible=False) as multi_row:
            ap_up  = gr.Image(type="pil", label="AP");      pa_up  = gr.Image(type="pil", label="PA")
            sky_up = gr.Image(type="pil", label="Skyline"); lat1_up= gr.Image(type="pil", label="Lat1")
            lat2_up= gr.Image(type="pil", label="Lat2")

        prompt_r = gr.Radio(
            ["Describe this X‑ray image.",
             "Generate the report (impression) of the following X‑ray images.",
             "Custom"],
            label="Prompt", value="Describe this X‑ray image."
        )
        custom_box = gr.Textbox(placeholder="Type your custom prompt …", visible=False)

        run = gr.Button("🔍 Run inference", variant="primary")

        with gr.Row():
            out_base = gr.Markdown()
            out_lora = gr.Markdown()
        out_img = gr.Image(type="pil", label="Image shown to model")

        # dynamic visibilities
        mode.change(lambda m: (gr.update(visible=m=="Single image"),
                               gr.update(visible=m!="Single image")),
                    mode, [single_up, multi_row])
        prompt_r.change(lambda p: gr.update(visible=p=="Custom"),
                        prompt_r, custom_box)

        run.click(generate,
                  inputs=[mode, single_up, ap_up, pa_up, sky_up, lat1_up, lat2_up,
                          prompt_r, custom_box,
                          st_base, st_ad, st_bf16, st_cpu],
                  outputs=[out_base, out_lora, out_img])

    return demo

# ───────────────────────── CLI / entry ‑ point ────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_id", default="google/medgemma-4b-it",
                    help="HuggingFace model id or local path of the *base* model")
    ap.add_argument("--adapter_dir",   required=True,
                    help="Directory containing LoRA adapter")
    ap.add_argument("--bf16", action="store_true",
                    help="Use BF16 (if GPU supports)")
    ap.add_argument("--cpu",  action="store_true",
                    help="Run both models on CPU")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    demo = build_demo(args.base_model_id, args.adapter_dir, args.bf16, args.cpu)
    demo.launch(server_port=args.port, share=args.share)

if __name__ == "__main__":
    main()
