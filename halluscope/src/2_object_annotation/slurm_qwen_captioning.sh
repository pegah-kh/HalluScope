#!/bin/bash

#SBATCH --job-name=qwen_captioning
#SBATCH --array=0-2
#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G  # each GPU process loads the ~19 GB model through CPU RAM; the 8 GB default gets OOM-killed
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out

set -e

ENV_DIR=/data/khayatan/envs/qwen35
COCO_DIR=/data/common/coco
CACHE_DIR=/data/khayatan/cache
BATCH_SIZE=16
NUM_SHARDS=3  # must match --array=0-2, and stay fixed when resubmitting only some tasks (e.g. --array=0,2)

export PATH=$ENV_DIR/bin:$PATH
cd "$SLURM_SUBMIT_DIR"

NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}

accelerate launch --num_processes=$NUM_GPUS --main_process_port 0 --mixed_precision=bf16 src/2_object_annotation/1_qwen_captioning.py \
    --coco_dir $COCO_DIR \
    --cache_dir $CACHE_DIR \
    --batch_size $BATCH_SIZE \
    --shard_id $SLURM_ARRAY_TASK_ID \
    --num_shards $NUM_SHARDS
