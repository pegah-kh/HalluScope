#!/bin/bash
# One pool of chosen/rejected pairs per pair type.
source "$(dirname "$0")/common.sh"

CONTRASTIVE_THRESHOLD=$([ "$MODEL" = qwen ] && echo 0.92 || echo 0.9)
A="$ANSWERS"

pairs() {  # filter type, chosen answers, rejected answers, threshold, pool name
    echo "$PYTHON 5_1_filter_pairs.py --model $MODEL --type $1 --chosen $2 --rejected $3 --threshold $4 --output_path $PAIRS/$5.json"
}

mkdir -p "$PAIRS"

JOBS=(
    "$(pairs cpq "$A/cpq_text_hint.json" "$A/cpq_text.json" 0.9 cpq)"
    "$(pairs tpq "$A/tpq_text.json" "$A/tpq_text_hint.json" 0.9 tpq)"
    "$(pairs pope "$A/pope_text.json" "$A/pope_text_hint.json" 0.9 pope)"
    "$(pairs pope_negative "$A/pope_negative_text.json" "$A/pope_negative_text_hint.json" 0.9 pope_negative)"
    "$(pairs description "$A/description_unilateral_text_hint.json" "$A/description_unilateral_text.json" 0.9 description_unilateral)"
    "$(pairs description "$A/description_contrastive_text.json" "$A/description_contrastive_text_hint.json" "$CONTRASTIVE_THRESHOLD" description_contrastive)"
    "$(pairs description "$A/description_unilateral_text.json" "$A/description_injection_text_hint.json" 0.95 description_injection)"
)
run_on_gpus

$PYTHON 5_2_attribute_pairs.py --attribute_answers "$A/attribute_answers.jsonl" --output_path "$PAIRS/attribute.json"
