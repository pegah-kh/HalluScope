# Adapted from https://github.com/Wu0409/Antidote/blob/main/data_pipeline/2_2_infer_grounding.py
import argparse
import json
import os

import groundingdino.datasets.transforms as T
import numpy as np
import torch
from groundingdino.models import build_model
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

from utils import read_jsonl

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.35
PRESENT_LOGIT_THRESHOLD = 0.45

TRANSFORM = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(config_path, repo_id="ShilongLiu/GroundingDINO", filename="groundingdino_swinb_cogcoor.pth"):
    model = build_model(SLConfig.fromfile(config_path))
    checkpoint = torch.load(hf_hub_download(repo_id=repo_id, filename=filename), map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    return model.eval()


def detect(model, image, objects):
    image_tensor, _ = TRANSFORM(image.convert("RGB"), None)
    _, logits, phrases = predict(model, image_tensor, " . ".join(objects), BOX_THRESHOLD, TEXT_THRESHOLD, device="cuda")
    return logits, phrases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Grounding-DINO on the present and absent objects of each image.")
    parser.add_argument("--triplets", required=True, help="Output of 1_triplets.py.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--groundingdino_config", required=True, help="GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    model = load_model(args.groundingdino_config)

    with open(args.output_path, "w") as f:
        for i, triplet in enumerate(tqdm(read_jsonl(args.triplets))):
            image = Image.open(os.path.join(args.image_dir, f"{i}.jpg"))

            logits, phrases = detect(model, image, triplet["present"])
            confident = np.where(logits > PRESENT_LOGIT_THRESHOLD)[0]
            present = [phrases[j] for j in confident]
            _, absent_detected = detect(model, image, triplet["no_exist"])

            detection = {"idx": i, "present": list(set(present)), "no_exist": list(set(absent_detected))}
            f.write(json.dumps(detection) + "\n")
