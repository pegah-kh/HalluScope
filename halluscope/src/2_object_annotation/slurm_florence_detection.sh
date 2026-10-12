#!/bin/bash
#SBATCH --job-name=florence_detection
#SBATCH --array=0-2
#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out

set -e

ENV_DIR=/data/khayatan/envs/dino  # transformers 4.x, Florence-2's remote code breaks on 5.x
COCO_DIR=/data/common/coco
CACHE_DIR=/data/khayatan/cache
BATCH_SIZE=16
NUM_SHARDS=3  # must match --array=0-2, and stay fixed when resubmitting only some tasks (e.g. --array=0,2)

cd "$SLURM_SUBMIT_DIR"

NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}

# Port 0 = pick a free port, so array tasks landing on the same node don't collide on 29500
$ENV_DIR/bin/accelerate launch --num_processes=$NUM_GPUS --main_process_port 0 --mixed_precision=fp16 src/2_object_annotation/2_florence_objs_detection.py \
    --coco_dir $COCO_DIR \
    --cache_dir $CACHE_DIR \
    --batch_size $BATCH_SIZE \
    --shard_id $SLURM_ARRAY_TASK_ID \
    --num_shards $NUM_SHARDS
