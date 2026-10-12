#!/bin/bash
# The target model answers every prompt list, plain (text) and hinted (text_hint).
# Then: hallucination-injection prompts are built from its descriptions and continued,
# and GPT writes wrong-attribute answers to its TPQ answers (needs OPENAI_API_KEY).
source "$(dirname "$0")/common.sh"

SHORT=128
LONG=512
GENERATE="$PYTHON 4_1_generate_answers.py --model $MODEL ${MODEL_PATH:+--model_path $MODEL_PATH} --image_root $IMAGES"

mkdir -p "$ANSWERS"

JOBS=()
for field in text text_hint; do
    for list in cpq tpq pope pope_negative; do
        JOBS+=("$GENERATE --prompts $PROMPTS/$list.jsonl --field $field --max_new_tokens $SHORT --output_path $ANSWERS/${list}_$field.json")
    done
    for list in description_unilateral description_contrastive; do
        JOBS+=("$GENERATE --prompts $PROMPTS/$list.jsonl --field $field --max_new_tokens $LONG --output_path $ANSWERS/${list}_$field.json")
    done
done
run_on_gpus

$PYTHON 4_2_hall_inject_prompts.py --verified "$DATA_ROOT/verified.jsonl" \
    --descriptions "$ANSWERS/description_unilateral_text.json" --model "$MODEL" \
    --output_path "$ANSWERS/description_injection_prompts.jsonl"
JOBS=("$GENERATE --prompts $ANSWERS/description_injection_prompts.jsonl --field text_hint --inject --max_new_tokens $LONG \
    --output_path $ANSWERS/description_injection_text_hint.json")
run_on_gpus

$PYTHON 4_3_attribute_answers.py --tpq_outputs "$ANSWERS/tpq_text.json" --model "$MODEL" \
    --output_path "$ANSWERS/attribute_answers.jsonl"
