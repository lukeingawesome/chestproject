#!/bin/bash

export CUDA_VISIBLE_DEVICES="4,5"

torchrun --standalone --nnodes 1 --nproc_per_node 2 \
         train_medgemma_knee.py --resume_from_checkpoint ./medgemma-4b-it-sft-lora-knee-ddp/checkpoint-30000