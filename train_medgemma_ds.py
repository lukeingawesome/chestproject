#!/usr/bin/env python
# ddp_medgemma.py
# Instruction‑tune MedGemma with one model replica per GPU (DDP).

import os, gc, sys
import argparse
from pathlib import Path
from typing import Any, List, Optional
from dataclasses import dataclass

import pandas as pd
import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoConfig,
    set_seed,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# ─────────────────────────────── DDP utilities ────────────────────────────
### DDP‑CHANGE: pick up rank information that torchrun/accelerate exports
LOCAL_RANK  = int(os.getenv("LOCAL_RANK",   0))
GLOBAL_RANK = int(os.getenv("RANK",         0))
WORLD_SIZE  = int(os.getenv("WORLD_SIZE",   1))

print(f"[rank {GLOBAL_RANK}] starting — local_rank={LOCAL_RANK} world_size={WORLD_SIZE}")

device = torch.device("cuda", LOCAL_RANK) if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.set_device(device)  # just in case
# -------------------------------------------------------------------------

@dataclass
class Sample:
    images:   List[Image.Image]
    messages: List[dict[str, Any]]

class StreamingMapDataset:
    def __init__(self, lazy):
        self.lazy = lazy
    def __len__(self):  return len(self.lazy)
    def __getitem__(self, i):
        s = self.lazy[i]
        return {} if s is None else {"images": s.images, "messages": s.messages}

class LazyImageDataset:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        if GLOBAL_RANK == 0:
            print(f"📊  {len(df)} rows loaded (lazy)")
    def __len__(self):  return len(self.df)

    # columns
    IMAGE_COLUMN    = "img_path"
    INSTR_COLUMN    = "instruction"
    TARGET_COLUMN   = "full_answer"

    # helpers ---------------------------------------------------------------
    def _load_img(self, p: str) -> Optional[Image.Image]:
        try:
            pth = Path(p)
            if not pth.exists(): return None
            img = Image.open(pth)
            return img.convert("RGB") if img.mode != "RGB" else img
        except Exception: return None

    def _prompt_and_images(self, row):
        content, imgs = [], []
        content.append({"type": "text", "text": "<task=report>"})

        img = self._load_img(row[self.IMAGE_COLUMN])
        if img is not None:
            imgs.append(img)
            content.append({"type": "image"})

        instr = str(row[self.INSTR_COLUMN]) if pd.notna(row[self.INSTR_COLUMN]) else ""
        if instr.strip():
            content.append({"type": "text", "text": instr.strip()})
        return content, imgs

    # main getter -----------------------------------------------------------
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        content, imgs = self._prompt_and_images(row)

        tgt = str(row[self.TARGET_COLUMN]) if pd.notna(row[self.TARGET_COLUMN]) else ""
        if not imgs or not tgt.strip():     # skip rows w/o img or answer
            return None

        return Sample(
            images   = imgs,
            messages = [
                {"role":"user",      "content": content},
                {"role":"assistant", "content":[{"type":"text", "text":tgt.strip()}]},
            ],
        )

# ─────────────────────────────── collator ─────────────────────────────────
class CollateFn:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        batch = [b for b in batch if b]
        texts, img_lists = [], []
        for ex in batch:
            texts.append(
                self.processor.apply_chat_template(
                    ex["messages"],
                    add_generation_prompt=False,
                    tokenize=False
                ).strip()
            )
            img_lists.append(ex["images"])
        out = self.processor(
            text=texts, images=img_lists,
            padding=True, return_tensors="pt"
        )
        labels = out["input_ids"].clone()
        special = [self.processor.tokenizer.pad_token_id]
        if getattr(self.processor, "image_token_id", None) is not None:
            special.append(self.processor.image_token_id)
        for t in special:
            labels[labels == t] = -100
        out["labels"] = labels
        return out

# ─────────────────────────────── evaluation callback ───────────────────────
class SampleLoggingCallback(TrainerCallback):
    def __init__(self, processor, val_dataset, num_samples=3):
        self.processor = processor
        self.val_dataset = val_dataset
        self.num_samples = num_samples
        self.sample_indices = list(range(min(num_samples, len(val_dataset))))
    
    def on_evaluate(self, args, state, control, model, **kwargs):
        if GLOBAL_RANK == 0:  # Only log on main process
            print("\n" + "="*80)
            print("🔍 SAMPLE VALIDATION OUTPUTS")
            print("="*80)
            
            model.eval()
            with torch.no_grad():
                for i, idx in enumerate(self.sample_indices):
                    if idx >= len(self.val_dataset):
                        break
                        
                    sample = self.val_dataset[idx]
                    if not sample:  # Skip None samples
                        continue
                        
                    # Get instruction and ground truth
                    instruction = ""
                    ground_truth = ""
                    
                    # Extract instruction from user message
                    for msg in sample["messages"]:
                        if msg["role"] == "user":
                            for content in msg["content"]:
                                if content["type"] == "text" and content["text"] != "<task=report>":
                                    instruction = content["text"]
                                    break
                    
                    # Extract ground truth from assistant message
                    for msg in sample["messages"]:
                        if msg["role"] == "assistant":
                            for content in msg["content"]:
                                if content["type"] == "text":
                                    ground_truth = content["text"]
                                    break
                    
                    # Generate prediction
                    inputs = self.processor(
                        text=[self.processor.apply_chat_template(
                            sample["messages"][:1],  # Only user message
                            add_generation_prompt=True,
                            tokenize=False
                        )],
                        images=[sample["images"]],
                        return_tensors="pt"
                    ).to(model.device)
                    
                    # Generate response
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                    )
                    
                    # Decode prediction
                    prediction = self.processor.tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1]:], 
                        skip_special_tokens=True
                    ).strip()
                    
                    print(f"\n📋 Sample {i+1}:")
                    print(f"Instruction: {instruction}")
                    print(f"Answer: {prediction}")
                    print(f"Ground Truth: {ground_truth}")
                    print("-" * 60)
            
            print("="*80 + "\n")
            model.train()

# ─────────────────────────────── config ───────────────────────────────────
model_id   = "google/medgemma-4b-it"
csv_path   = "/data3/public/chest/foundation_v1.0.csv"
output_dir = "medgemma-4b-it-sft-lora-chest-ddp"

epochs          = 1
lr              = 2e-4
per_gpu_batch   = 2          # each process sees this batch size
grad_accum      = 2
seed            = 42

# ─────────────────────────────── helpers ──────────────────────────────────
def load_datasets():
    df = pd.read_csv(csv_path)
    assert "split" in df.columns, "CSV needs a 'split' column"
    train = df[df["split"] == "train"].copy()
    val   = df[df["split"] == "val"  ].copy() if "val"  in df.split.unique() else pd.DataFrame()
    train_set = StreamingMapDataset(LazyImageDataset(train))
    val_set   = StreamingMapDataset(LazyImageDataset(val)) if len(val) else None
    return train_set, val_set

def parse_args():
    parser = argparse.ArgumentParser(description="Train MedGemma model with DDP and resume support")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint directory to resume from")
    return parser.parse_args()

# ─────────────────────────────── main ─────────────────────────────────────
def main():
    # Parse command line arguments
    args = parse_args()
    
    if GLOBAL_RANK == 0:
        print("🚀 Starting DDP fine‑tune")

    set_seed(seed)

    # datasets
    train_ds, val_ds = load_datasets()

    # model (+ optionally keep tower on CPU)
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(LOCAL_RANK)[0] >= 8
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
    ).to(device)

    # keep the SigLIP tower on CPU if VRAM is tight
    model.vision_tower.to("cpu")
    for p in model.vision_tower.parameters():
        p.requires_grad = False

    # LoRA
    lora_cfg = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj","lm_head"],
        task_type="CAUSAL_LM", modules_to_save=["embed_tokens","lm_head"],
    )
    model = get_peft_model(model, lora_cfg)

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.model_max_length = 2048
    processor.tokenizer.padding_side = "right"

    # collate function
    collate_fn = CollateFn(processor)

    # Create evaluation callback for sample logging
    callbacks = []
    if val_ds:
        sample_callback = SampleLoggingCallback(processor, val_ds, num_samples=3)
        callbacks.append(sample_callback)

    # trainer config
    args_config = SFTConfig(
        output_dir                  = output_dir,
        num_train_epochs            = epochs,
        per_device_train_batch_size = per_gpu_batch,
        per_device_eval_batch_size  = per_gpu_batch,
        gradient_accumulation_steps = grad_accum,
        learning_rate               = lr,
        bf16                        = use_bf16,
        gradient_checkpointing      = False,
        logging_steps               = 100,
        save_strategy               = "steps",
        save_steps                  = 10000,
        eval_strategy               = "steps" if val_ds else "no",
        eval_steps                  = 10000,
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
        args          = args_config,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        data_collator = collate_fn,
        tokenizer     = processor.tokenizer,
        callbacks     = callbacks,
    )

    if GLOBAL_RANK == 0:
        t    = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🔍 Trainable = {trainable:,} / {t:,} "
              f"({100*trainable/t:.2f} %)")

    # Resume training if checkpoint provided
    if args.resume_from_checkpoint:
        if GLOBAL_RANK == 0:
            print(f"🔄 Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
        
    if GLOBAL_RANK == 0:
        print("✅ Training finished, saving …")
    trainer.save_model()

    # housekeeping
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
