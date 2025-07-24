#!/usr/bin/env python
# ddp_medgemma.py – refined
# Fine‑tune google/medgemma‑4b‑it on CXR reports with DDP‑LoRA.

import os, gc, argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Optional

import pandas as pd
import torch
from PIL import Image
from transformers import (
    AutoProcessor, AutoModelForImageTextToText,
    set_seed, TrainerCallback
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# ──────────────────────────────────────────────────────────────────────────
#                          distributed environment
# ──────────────────────────────────────────────────────────────────────────
LOCAL_RANK  = int(os.getenv("LOCAL_RANK",   0))
GLOBAL_RANK = int(os.getenv("RANK",         0))
WORLD_SIZE  = int(os.getenv("WORLD_SIZE",   1))
print(f"[rank {GLOBAL_RANK}] launch — local_rank={LOCAL_RANK} world_size={WORLD_SIZE}")

device = torch.device("cuda", LOCAL_RANK) if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(device)

# ──────────────────────────────────────────────────────────────────────────
#                               configuration
# ──────────────────────────────────────────────────────────────────────────
class CFG:
    model_id      = "google/medgemma-4b-it"
    csv_path      = "/data3/public/chest/foundation_v1.0.csv"
    out_dir       = "medgemma-4b-it-sft-lora-chest-ddp"
    seed          = 42
    epochs        = 2             # train for ≥2 epochs or use early‑stop
    lr            = 2e-4
    per_gpu_bs    = 2
    grad_accum    = 2
    lora_r        = 16
    lora_dropout  = 0.05
    log_steps     = 100
    save_steps    = 10_000

# ──────────────────────────────────────────────────────────────────────────
#                              data classes
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Sample:
    images:   List[Image.Image]
    messages: List[dict[str, Any]]        # ChatML‐like dicts

class LazyImageDataset:
    """Lazily load images to keep RAM usage low."""
    IMAGE_COL  = "img_path"
    INSTR_COL  = "instruction"
    TARGET_COL = "full_answer"

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        if GLOBAL_RANK == 0:
            print(f"📊  {len(self.df)} rows loaded (lazy)")

    def __len__(self): return len(self.df)

    # --------------- helpers ----------------
    @staticmethod
    def _open_rgb(path: str) -> Optional[Image.Image]:
        try:
            img = Image.open(Path(path))
            return img.convert("RGB")
        except Exception:
            return None

    def __getitem__(self, idx: int) -> Optional[Sample]:
        row = self.df.iloc[idx]
        img  = self._open_rgb(row[self.IMAGE_COL])
        tgt  = str(row[self.TARGET_COL]).strip()

        if img is None or not tgt:
            return None                       # drop bad / empty rows

        instr = str(row[self.INSTR_COL]).strip()
        content = [
            {"type": "text", "text": "<task=report>"},
            {"type": "image"},
        ]
        if instr:
            content.append({"type": "text", "text": instr})

        return Sample(
            images   = [img],
            messages = [
                {"role": "user",  "content": content},
                {"role": "model", "content": [{"type": "text", "text": tgt}]},
            ]
        )

class StreamingMapDataset:
    """Wraps the lazy dataset in a Map‑style dataset that can drop Nones."""
    def __init__(self, lazy): self.lazy = lazy
    def __len__(self): return len(self.lazy)
    def __getitem__(self, i):
        s = self.lazy[i]
        return {} if s is None else {"images": s.images, "messages": s.messages}

# ──────────────────────────────────────────────────────────────────────────
#                             collate function
# ──────────────────────────────────────────────────────────────────────────
class CollateFn:
    """Tokenise + build image tensors + MASK the prompt."""
    def __init__(self, processor):
        self.processor = processor
        tok = processor.tokenizer
        self.pad_id       = tok.pad_token_id
        self.start_id     = tok.convert_tokens_to_ids("<start_of_turn>")
        self.end_id       = tok.convert_tokens_to_ids("<end_of_turn>")
        self.model_id     = tok.convert_tokens_to_ids("model")

    def _mask_prompt(self, input_ids):
        """Return labels with user / system tokens masked to −100."""
        labels = input_ids.clone()
        for row in range(labels.size(0)):
            ids = labels[row]
            # locate second "<start_of_turn>" (assistant turn)
            starts = (ids == self.start_id).nonzero(as_tuple=True)[0]
            if len(starts) < 2:                       # safety
                labels[row].fill_(-100)
                continue
            ass_start = starts[1]                     # index of token
            # mask every token *up to and incl. 'model' token*
            labels[row, :ass_start + 2] = -100        # +2 covers "<sot>,model"
            # also mask padding & image tokens
            labels[row, ids == self.pad_id] = -100
            if hasattr(self.processor, "image_token_id"):
                labels[row, ids == self.processor.image_token_id] = -100
        return labels

    def __call__(self, batch):
        batch = [b for b in batch if b]               # drop {} placeholders
        chat_texts = [
            self.processor.apply_chat_template(
                ex["messages"], add_generation_prompt=False, tokenize=False
            ).strip()
            for ex in batch
        ]
        img_lists = [ex["images"] for ex in batch]

        out = self.processor(
            text   = chat_texts,
            images = img_lists,
            padding=True,
            return_tensors="pt"
        )
        out["labels"] = self._mask_prompt(out["input_ids"])
        return out

# ──────────────────────────────────────────────────────────────────────────
#                         validation / logging callback
# ──────────────────────────────────────────────────────────────────────────
class SampleLoggingCallback(TrainerCallback):
    """Print a few deterministic generations every eval step."""
    def __init__(self, processor, val_ds, n=3):
        self.pro = processor
        self.val = val_ds
        self.n   = min(n, len(val_ds))

    def on_evaluate(self, args, state, control, model, **kwargs):
        if GLOBAL_RANK != 0:                          # only once
            return
        print("\n" + "="*80)
        model.eval()
        with torch.no_grad():
            for i in range(self.n):
                sample = self.val[i]
                if not sample: continue
                prompt = self.pro.apply_chat_template(
                    sample["messages"][:1],           # user part only
                    add_generation_prompt=True,
                    tokenize=False
                )
                inputs = self.pro(
                    text=[prompt], images=[sample["images"]],
                    return_tensors="pt"
                ).to(model.device)

                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    eos_token_id = self.pro.tokenizer.convert_tokens_to_ids(
                        "<end_of_turn>"
                    ),
                    pad_token_id = self.pro.tokenizer.pad_token_id,
                    do_sample=False,                  # deterministic
                    no_repeat_ngram_size=4,
                    repetition_penalty=1.1,
                )
                answer = self.pro.tokenizer.decode(
                    gen_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                truth = sample["messages"][1]["content"][0]["text"]
                print(f"\n📋 Sample {i+1}")
                print(f"Answer       : {answer}")
                print(f"Ground Truth : {truth}")
                print("-"*60)
        print("="*80 + "\n")
        model.train()

# ──────────────────────────────────────────────────────────────────────────
#                               utilities
# ──────────────────────────────────────────────────────────────────────────
def load_splits():
    df = pd.read_csv(CFG.csv_path)
    assert "split" in df.columns, "CSV must contain a 'split' column"
    train = df[df.split == "train"]
    val   = df[df.split == "val"]   if "val" in df.split.unique() else pd.DataFrame()
    return (
        StreamingMapDataset(LazyImageDataset(train)),
        StreamingMapDataset(LazyImageDataset(val)) if len(val) else None,
    )

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume_from_checkpoint")
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────────
#                                   main
# ──────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    if GLOBAL_RANK == 0:
        print("🚀  Starting MedGemma LoRA fine‑tune")
    set_seed(CFG.seed)

    # data
    train_ds, val_ds = load_splits()

    # model + LoRA
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    model = AutoModelForImageTextToText.from_pretrained(
        CFG.model_id, torch_dtype=torch.bfloat16 if bf16 else torch.float32
    ).to(device)

    # ‑‑ keeping vision tower on GPU speeds training; comment these two
    #     lines out only if VRAM is too tight
    # model.vision_tower.to("cpu")
    # for p in model.vision_tower.parameters(): p.requires_grad = False

    lora_cfg = LoraConfig(
        r              = CFG.lora_r,
        lora_alpha     = CFG.lora_r,
        lora_dropout   = CFG.lora_dropout,
        target_modules = [
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj","lm_head"
        ],
        bias           = "none",
        task_type      = "CAUSAL_LM",
        modules_to_save= ["embed_tokens","lm_head"],
    )
    model = get_peft_model(model, lora_cfg)

    # processor / collator
    processor  = AutoProcessor.from_pretrained(CFG.model_id)
    processor.tokenizer.model_max_length = 2048
    processor.tokenizer.padding_side    = "right"
    collate_fn = CollateFn(processor)

    # trainer
    callbacks = []
    if val_ds: callbacks.append(SampleLoggingCallback(processor, val_ds, n=3))

    trainer_cfg = SFTConfig(
        output_dir                  = CFG.out_dir,
        num_train_epochs            = CFG.epochs,
        per_device_train_batch_size = CFG.per_gpu_bs,
        per_device_eval_batch_size  = CFG.per_gpu_bs,
        gradient_accumulation_steps = CFG.grad_accum,
        learning_rate               = CFG.lr,
        bf16                        = bf16,
        logging_steps               = CFG.log_steps,
        save_strategy               = "steps",
        save_steps                  = CFG.save_steps,
        eval_strategy         = "steps" if val_ds else "no",
        eval_steps                  = CFG.save_steps,
        optim                       = "adamw_torch_fused",
        warmup_ratio                = 0.03,
        max_grad_norm               = 0.3,
        lr_scheduler_type           = "linear",
        remove_unused_columns       = False,
        dataset_kwargs              = {"skip_prepare_dataset": True},
        label_names                 = ["labels"],
        save_total_limit            = 1,
        save_on_each_node           = True,
        dataloader_num_workers      = 4,
        dataloader_pin_memory       = True,
    )

    trainer = SFTTrainer(
        model         = model,
        args          = trainer_cfg,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        data_collator = collate_fn,
        callbacks     = callbacks,
    )

    if GLOBAL_RANK == 0:
        tot = sum(p.numel() for p in model.parameters())
        trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🔧  Trainable params: {trn:,} / {tot:,} "
              f"({100*trn/tot:.2f} %)")

    # train / resume
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if GLOBAL_RANK == 0:
        print("✅  Training finished – saving adapter")
    trainer.save_model()

    # housekeeping
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
