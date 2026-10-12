#!/bin/bash
# Captions -> GPT triplets -> SD3 images -> Grounding-DINO verification. Needs OPENAI_API_KEY and HF_TOKEN.
source "$(dirname "$0")/common.sh"

mkdir -p "$DATA_ROOT" "$IMAGES"

$PYTHON 0_caption_pool.py --cc3m_file "$CC3M_JSON" --concreteness_csv "$CONCRETENESS_CSV" \
    --target_count "$NUM_CAPTIONS" --output_path "$DATA_ROOT/captions.json"
$PYTHON 1_triplets.py --caption_pool "$DATA_ROOT/captions.json" --total_num "$NUM_CAPTIONS" \
    --output_path "$DATA_ROOT/triplets.jsonl"
$PYTHON 2_1_generate_images.py --triplets "$DATA_ROOT/triplets.jsonl" --save_dir "$IMAGES" --num_gpus "$NUM_GPUS"
$PYTHON 2_2_grounding.py --triplets "$DATA_ROOT/triplets.jsonl" --image_dir "$IMAGES" \
    --groundingdino_config "$GROUNDINGDINO_CONFIG" --output_path "$DATA_ROOT/detections.jsonl"
$PYTHON 2_3_filter_detections.py --triplets "$DATA_ROOT/triplets.jsonl" --detections "$DATA_ROOT/detections.jsonl" \
    --output_path "$DATA_ROOT/verified.jsonl"
