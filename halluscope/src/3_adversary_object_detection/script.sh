#!/bin/bash
# Runs on the current machine. For several machines, use slurm_adversary_discovery.sh instead.
set -e
cd "$(dirname "$0")/../.."

QWEN_ENV_DIR=/data/khayatan/envs/qwen35  # transformers 5.x, needed for Qwen3.5
COCO_DIR=/data/common/coco
CACHE_DIR=/data/khayatan/cache

NUM_GPUS=$(nvidia-smi -L | wc -l)


# FOR FLORENCE OBJECTS (3_1) AND INSTANCES OBJECTS (3_2) SEPARATELY:
for SOURCE in florence instances; do

    # 1. CONTEXT-AWARE ADVERSARY DISCOVERY (on all GPUs of this machine).
    # Reads data_files/${SOURCE}_dino_filtered_objects.json,
    # writes data_files/coco_${SOURCE}_adversary_random_present_objects.json

    # Port 0 = pick a free port, so it doesn't collide with other runs on the same machine
    $QWEN_ENV_DIR/bin/accelerate launch --num_processes=$NUM_GPUS --main_process_port 0 src/3_adversary_object_detection/1_context_aware_adversary_discovery.py \
        --coco_dir $COCO_DIR \
        --cache_dir $CACHE_DIR \
        --objects_file data_files/${SOURCE}_dino_filtered_objects.json \
        --objects_name coco_${SOURCE}_objects

    # With several GPUs, each one writes its own part file; merge them
    # (with a single GPU the script already wrote the final file itself)
    if [ $NUM_GPUS -gt 1 ]; then
        $QWEN_ENV_DIR/bin/python src/3_adversary_object_detection/1_context_aware_adversary_discovery.py \
            --objects_name coco_${SOURCE}_objects --merge --num_parts $NUM_GPUS
    fi


    # 2. KEEPING ONLY SAMPLES WITH MORE THAN 3 PRESENT OBJECTS.
    # Writes data_files/coco_${SOURCE}_adversary_random_present_objects_filtered.json

    $QWEN_ENV_DIR/bin/python src/3_adversary_object_detection/2_filtering_samples.py \
        --objects_file data_files/${SOURCE}_dino_filtered_objects.json \
        --objects_name coco_${SOURCE}_objects

done
