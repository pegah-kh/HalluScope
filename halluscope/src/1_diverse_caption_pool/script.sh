#!/bin/bash
cd "$(dirname "$0")/../.."

COCO_DIR=/data/common/coco

# 1. SELECT A DIVERSE POOL OF 30K COCO CAPTIONS VIA K-CENTER GREEDY SELECTION.
# Outputs go to data_files/selected_coco_indices_30k.npy and data_files/selected_coco_metadata.json
python src/1_diverse_caption_pool/create_diverse_caption_pool.py \
    --coco_dir $COCO_DIR \
    --num_samples 30000 \
    --embedding_model all-MiniLM-L6-v2
