#!/bin/bash
# Evaluates a vision-language model on the HalluScope benchmark (benchmark/v1: the paper's test set, benchmark/v2: regenerated).
# Can be run from anywhere: moves to the project root (halluscope/halluscope/) first.
set -e
cd "$(dirname "$0")/../.."

MODEL=Qwen/Qwen2.5-VL-7B-Instruct  # any Hugging Face chat VLM that loads with AutoModelForImageTextToText
VERSION=v2                          # benchmark version: v1 (paper) or v2 (regenerated)
RUN_NAME=$(basename $MODEL)_$VERSION  # name of the predictions / scores folders
COCO_DIR=/path/to/coco              # COCO root folder, must contain train2017/
CACHE_DIR=                          # Hugging Face cache folder; empty = Hugging Face's default


# 1. ANSWER EVERY TEST QUESTION WITH THE MODEL (one GPU, greedy decoding).
# Writes data_files/predictions/$RUN_NAME/{task}_{subset}.jsonl
# Add --limit 16 for a quick check on the first 16 questions of each test set.

python src/evaluation/run_inference.py \
    --model $MODEL \
    --version $VERSION \
    --run_name $RUN_NAME \
    --coco_dir $COCO_DIR \
    ${CACHE_DIR:+--cache_dir $CACHE_DIR}  # only passed when CACHE_DIR is set


# 2. RECOGNITION (YES/NO) QUESTIONS: per-test-set accuracy and POPE-style precision / recall / F1.
# Writes data_files/scores/$RUN_NAME/recognition.json

python src/evaluation/evaluate_recognition.py --run_name $RUN_NAME


# 3. ADVERSARY-PRESUPPOSITION QUESTIONS, JUDGED BY AN LLM.
# Writes data_files/scores/$RUN_NAME/presupposition.json
# With OpenAI (the paper's judge, gpt-5-mini): needs the OPENAI_API_KEY environment variable.

python src/evaluation/evaluate_presupposition.py --run_name $RUN_NAME

# Or with a local open judge behind an OpenAI-compatible server, e.g. started with
#   vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8000
# python src/evaluation/evaluate_presupposition.py --run_name $RUN_NAME \
#     --api_base http://localhost:8000/v1 --judge_model meta-llama/Llama-3.3-70B-Instruct
