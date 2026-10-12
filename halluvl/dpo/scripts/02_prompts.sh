#!/bin/bash
# Queries with and without hints. Needs OPENAI_API_KEY (CPQ/TPQ questions).
source "$(dirname "$0")/common.sh"

mkdir -p "$PROMPTS"

$PYTHON 3_1_cpq_tpq_questions.py --verified "$DATA_ROOT/verified.jsonl" --output_path "$PROMPTS/cpq_tpq_questions.jsonl"
$PYTHON 3_2_cpq_tpq_prompts.py --questions "$PROMPTS/cpq_tpq_questions.jsonl" \
    --output_tpq "$PROMPTS/tpq.jsonl" --output_cpq "$PROMPTS/cpq.jsonl"
$PYTHON 3_3_pope_prompts.py --verified "$DATA_ROOT/verified.jsonl" --output_path "$PROMPTS/pope.jsonl"
$PYTHON 3_3_pope_prompts.py --verified "$DATA_ROOT/verified.jsonl" --output_path "$PROMPTS/pope_negative.jsonl" --negative
$PYTHON 3_4_description_prompts.py --verified "$DATA_ROOT/verified.jsonl" \
    --output_unilateral "$PROMPTS/description_unilateral.jsonl" --output_contrastive "$PROMPTS/description_contrastive.jsonl"
