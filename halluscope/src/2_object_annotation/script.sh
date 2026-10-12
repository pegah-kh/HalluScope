#!/bin/bash
# Runs on the current machine. For several machines, use slurm_qwen_captioning.sh instead.
set -e
cd "$(dirname "$0")/../.."

QWEN_ENV_DIR=/data/khayatan/envs/qwen35  # transformers 5.x, needed for Qwen3.5
FLORENCE_ENV_DIR=/data/khayatan/envs/dino  # transformers 4.x, Florence-2's remote code breaks on 5.x
COCO_DIR=/data/common/coco
CACHE_DIR=/data/khayatan/cache
BATCH_SIZE=16


# 1. QWEN CAPTIONING (on all GPUs of this machine).
# Reads data_files/selected_coco_metadata.json, writes data_files/qwen_captions.json

NUM_GPUS=$(nvidia-smi -L | wc -l)

# Port 0 = pick a free port, so it doesn't collide with other runs on the same machine
$QWEN_ENV_DIR/bin/accelerate launch --num_processes=$NUM_GPUS --main_process_port 0 --mixed_precision=bf16 src/2_object_annotation/1_qwen_captioning.py \
    --coco_dir $COCO_DIR \
    --cache_dir $CACHE_DIR \
    --batch_size $BATCH_SIZE

# With several GPUs, each one writes its own part file; merge only this run's parts
# (check first: "> qwen_captions.json" empties the file even when cat then fails)
if [ $NUM_GPUS -gt 1 ]; then
    PARTS=$(ls data_files/qwen_captions_part*of$(printf %03d $NUM_GPUS).json)
    cat $PARTS > data_files/qwen_captions.json
fi


# 2. FLORENCE OBJECT ANNOTATION (on all GPUs of this machine).
# Reads data_files/selected_coco_metadata.json, writes data_files/coco_florence_objects.json

$FLORENCE_ENV_DIR/bin/accelerate launch --num_processes=$NUM_GPUS --main_process_port 0 --mixed_precision=fp16 src/2_object_annotation/2_florence_objs_detection.py \
    --coco_dir $COCO_DIR \
    --cache_dir $CACHE_DIR \
    --batch_size $BATCH_SIZE

# With several GPUs, each one writes its own part file; merge them into coco_florence_objects.json
# (with a single GPU the script already wrote the final file itself)
if [ $NUM_GPUS -gt 1 ]; then
    $FLORENCE_ENV_DIR/bin/python src/2_object_annotation/2_florence_objs_detection.py --merge --num_parts $NUM_GPUS
fi


# 3_1. GROUNDING DINO + QWEN FILTERING OF THE FLORENCE OBJECTS (on all GPUs of this machine).
# Reads data_files/coco_florence_objects.json and data_files/qwen_captions.json,
# writes data_files/florence_dino_filtered_objects.json

$QWEN_ENV_DIR/bin/accelerate launch --num_processes=$NUM_GPUS --main_process_port 0 src/2_object_annotation/3_1_florence_grounding_dino_filtering.py \
    --coco_dir $COCO_DIR \
    --cache_dir $CACHE_DIR

# With several GPUs, each one writes its own part file; merge them into florence_dino_filtered_objects.json
# (with a single GPU the script already wrote the final file itself)
if [ $NUM_GPUS -gt 1 ]; then
    $QWEN_ENV_DIR/bin/python src/2_object_annotation/3_1_florence_grounding_dino_filtering.py --merge --num_parts $NUM_GPUS
fi


# 3_2. GROUNDING DINO + QWEN FILTERING OF THE COCO INSTANCES ANNOTATIONS (on all GPUs of this machine).
# Reads COCO's instances_train2017.json and data_files/qwen_captions.json,
# writes data_files/instances_dino_filtered_objects.json

$QWEN_ENV_DIR/bin/accelerate launch --num_processes=$NUM_GPUS --main_process_port 0 src/2_object_annotation/3_2_instances_grounding_dino_filtering.py \
    --coco_dir $COCO_DIR \
    --cache_dir $CACHE_DIR

# With several GPUs, each one writes its own part file; merge them into instances_dino_filtered_objects.json
# (with a single GPU the script already wrote the final file itself)
if [ $NUM_GPUS -gt 1 ]; then
    $QWEN_ENV_DIR/bin/python src/2_object_annotation/3_2_instances_grounding_dino_filtering.py --merge --num_parts $NUM_GPUS
fi