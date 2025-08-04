#!/usr/bin/env python
# inference_medgemma.py
# Generate MedGemma‑4B‑IT LoRA predictions for the *val* split.

import os, gc, argparse, json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Optional

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoProcessor, AutoModelForImageTextToText, set_seed
)
from peft import PeftConfig, PeftModel

# ──────────────────────────────────────────────────────────────────────────
#                               CLI arguments
# ──────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path",   required=True,
                   help="CSV containing a `split` column with 'val' rows.")
    p.add_argument("--adapter_dir", required=True,
                   help="Directory that holds the LoRA adapter saved by Trainer.")
    p.add_argument("--out_csv",    required=True,
                   help="Path where the CSV with predictions will be written.")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Images per forward pass (default 1 ‑ matches training).")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────
#                              data classes
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Sample:
    images:   List[Image.Image]
    messages: List[dict[str, Any]]

class LazyImageDataset:
    """Exactly the same lazy loader you used for training."""
    IMAGE_COL  = "img_path"
    INSTR_COL  = "instruction"
    TARGET_COL = "answer"          # not used in inference, but retained

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=False)  # keep original index
        print(f"📊  {len(self.df)} validation rows loaded")

    def __len__(self):  return len(self.df)

    # --------------- helpers ----------------
    @staticmethod
    def _open_rgb(path: str) -> Optional[Image.Image]:
        try:
            img = Image.open(Path(path))
            return img.convert("RGB")
        except Exception:
            return None

    def __getitem__(self, idx: int) -> Optional[dict[str, Any]]:
        row = self.df.iloc[idx]
        img = self._open_rgb(row[self.IMAGE_COL])
        if img is None:
            return None

        instr_txt = str(row[self.INSTR_COL]).strip()
        # build ChatML messages exactly like in training (but no answer yet)
        content = [
            {"type": "text", "text": "<task=report>"},
            {"type": "image"},
        ]
        if instr_txt:
            content.append({"type": "text", "text": instr_txt})

        messages = [
            {"role": "user", "content": content},
        ]
        return {
            "row_idx": row["index"],     # original df index for merging later
            "images":  [img],
            "messages": messages,
        }

class CollateFn:
    """Tokenise + build image tensors (no label masking needed for inference)."""
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        batch = [b for b in batch if b]                 # drop empty rows
        texts = [
            self.processor.apply_chat_template(
                ex["messages"],
                add_generation_prompt=True,             # important for inference
                tokenize=False
            ).strip()
            for ex in batch
        ]
        img_lists = [ex["images"] for ex in batch]
        model_inputs = self.processor(
            text=texts,
            images=img_lists,
            padding=True,
            return_tensors="pt"
        )
        model_inputs["row_idxs"] = [ex["row_idx"] for ex in batch]
        return model_inputs


# ──────────────────────────────────────────────────────────────────────────
#                                   main
# ──────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    # ── Select device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # ── Prepare data
    full_df = pd.read_csv(args.csv_path)
    assert "split" in full_df.columns, "CSV must contain a 'split' column"
    val_df  = full_df[full_df.split == "test"].copy()
    ds      = LazyImageDataset(val_df)
    # DataLoader with batch_size (still safe because images are small)
    from torch.utils.data import DataLoader
    collate_fn = CollateFn(
        AutoProcessor.from_pretrained(PeftConfig.from_pretrained(
            args.adapter_dir).base_model_name_or_path)
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # ── Load model + LoRA weights
    print("🔄  Loading base model + LoRA adapter …")
    peft_cfg      = PeftConfig.from_pretrained(args.adapter_dir)
    base_model    = AutoModelForImageTextToText.from_pretrained(
        peft_cfg.base_model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    # for inference we can merge LoRA into the base weights and drop adapters
    model = model.merge_and_unload()
    model.to(device).eval()

    processor = collate_fn.processor  # re‑use
    eos_id    = processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")

    # ── Iterate & generate
    predictions = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating predictions", unit="batch"):
            row_idxs = batch.pop("row_idxs")
            batch = {k: v.to(device) for k, v in batch.items()}
            gen_ids = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=eos_id,
                pad_token_id=processor.tokenizer.pad_token_id,
                do_sample=False,
                no_repeat_ngram_size=4,
                repetition_penalty=1.1,
            )

            # Decode only the newly generated tokens
            input_lens = batch["input_ids"].shape[1]
            for ridx, g in zip(row_idxs, gen_ids):
                text = processor.tokenizer.decode(
                    g[input_lens:], skip_special_tokens=True
                ).strip()
                predictions[ridx] = text

    # ── Merge predictions back into the original *val* dataframe
    val_df["prediction"] = val_df.index.map(predictions.get)
    # Optional: keep the ground‑truth column next to prediction for inspection
    if "answer" in val_df.columns:
        val_df = val_df[["img_path", "instruction", "answer", "prediction"]]
    else:
        val_df = val_df[["img_path", "instruction", "prediction"]]

    # ── Save
    val_df.to_csv(args.out_csv, index=False)
    print(f"✅  Saved {len(val_df)} predictions to: {args.out_csv}")

    # ── House‑keeping
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
