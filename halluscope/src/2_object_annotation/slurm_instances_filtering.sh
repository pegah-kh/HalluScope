#!/bin/bash
#SBATCH --job-name=instances_filtering
#SBATCH --array=0-2
#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G  # each GPU process loads the ~19 GB Qwen model through CPU RAM
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out

set -e

ENV_DIR=/data/khayatan/envs/qwen35  # transformers 5.x, needed for Qwen3.5
COCO_DIR=/data/common/coco
CACHE_DIR=/data/khayatan/cache
NUM_SHARDS=3  # must match --array=0-2, and stay fixed when resubmitting only some tasks (e.g. --array=0,2)

cd "$SLURM_SUBMIT_DIR"

NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}

# Port 0 = pick a free port, so array tasks landing on the same node don't collide on 29500
$ENV_DIR/bin/accelerate launch --num_processes=$NUM_GPUS --main_process_port 0 src/2_object_annotation/3_2_instances_grounding_dino_filtering.py \
    --coco_dir $COCO_DIR \
    --cache_dir $CACHE_DIR \
    --shard_id $SLURM_ARRAY_TASK_ID \
    --num_shards $NUM_SHARDS
