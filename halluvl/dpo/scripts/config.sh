# Edit this file (or export the same variables), then run the numbered scripts in order (see ../README.md).

# Where everything is written: captions, images, prompts, answers, pairs and the final dataset.
DATA_ROOT=${DATA_ROOT:-/path/to/halluvl_data}

# Inputs
CC3M_JSON=${CC3M_JSON:-/path/to/cc3m_train.json}                     # JSON list of {"caption": ...}
CONCRETENESS_CSV=${CONCRETENESS_CSV:-/path/to/concreteness_ratings.csv} # Brysbaert et al. 2014 (columns Word, Conc.M)
GROUNDINGDINO_CONFIG=${GROUNDINGDINO_CONFIG:-/path/to/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py}
NUM_CAPTIONS=${NUM_CAPTIONS:-30000}

# Target model whose answers are paired: qwen (Qwen2-VL-7B-Instruct) or llava (LLaVA-1.5-7B)
MODEL=${MODEL:-qwen}
# Optional: local path or Hub id replacing the default weights of MODEL
MODEL_PATH=${MODEL_PATH:-}

NUM_GPUS=${NUM_GPUS:-1}
PYTHON=${PYTHON:-python}
