#!/bin/bash
# Usage, from the project root (halluscope/halluscope/):
#   sbatch src/3_adversary_object_detection/slurm_adversary_discovery.sh florence
#   sbatch src/3_adversary_object_detection/slurm_adversary_discovery.sh instances
#SBATCH --job-name=adversary_discovery
#SBATCH --array=0-2
#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G  # each GPU process loads the ~19 GB Qwen model through CPU RAM
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out

set -e

SOURCE=${1:?usage: sbatch slurm_adversary_discovery.sh florence|instances}
ENV_DIR=/data/khayatan/envs/qwen35  # transformers 5.x, needed for Qwen3.5
COCO_DIR=/data/common/coco
CACHE_DIR=/data/khayatan/cache
NUM_SHARDS=3  # must match --array=0-2, and stay fixed when resubmitting only some tasks (e.g. --array=0,2)

cd "$SLURM_SUBMIT_DIR"

NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}

# Port 0 = pick a free port, so array tasks landing on the same node don't collide on 29500
$ENV_DIR/bin/accelerate launch --num_processes=$NUM_GPUS --main_process_port 0 src/3_adversary_object_detection/1_context_aware_adversary_discovery.py \
    --coco_dir $COCO_DIR \
    --cache_dir $CACHE_DIR \
    --objects_file data_files/${SOURCE}_dino_filtered_objects.json \
    --objects_name coco_${SOURCE}_objects \
    --shard_id $SLURM_ARRAY_TASK_ID \
    --num_shards $NUM_SHARDS
