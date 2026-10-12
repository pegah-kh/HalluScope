#!/bin/bash
# <model>_full.json (all pairs) and <model>_mixture.json (training set, CPQ:TPQ:POPE:description:attribute = 5:5:6:9:5).
source "$(dirname "$0")/common.sh"

$PYTHON 6_build_dataset.py --pairs_dir "$PAIRS" --model "$MODEL" --total_samples 20000 --output_dir "$DATASET"
