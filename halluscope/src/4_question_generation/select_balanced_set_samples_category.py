"""
Selects the final HalluScope test set: a diverse, category-balanced subset of the images with an adversary question.

Each adversary question is described by its intent (color, material, position, ..., found by keyword patterns)
and its absent object; both are embedded, the embeddings are weighted and concatenated, and K-Center Greedy
selection picks --num_samples images that cover these (intent, object) combinations as evenly as possible.
The four question sets of the selected images form the test set.

Reads, per subset (florence, instances), in data_files/:
  coco_{subset}_adversary_random_present_objects_filtered.json  (stage 3)
  coco_{subset}_difficult_samples_adversary_questions.json       (generate_adversary_questions.py)
  {subset}_pope_questions{,_negative,_random}.json               (generate_pope_questions.py)
Writes, in data_files/test_dataset/ (JSON arrays of {image, present_object, absent_object, random_object, text}):
  {adversary_presupposition,real_recognition,adversary_recognition,random_recognition}_{subset}_balanced_w_category.json
"""
import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm


# halluscope/halluscope/, two levels above this file's folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

QUESTION_INTENT_PATTERNS = {
    "color": [
        "what color", "what colour"
    ],
    "material": [
        "what material", "made of", "made from"
    ],
    "style": [
        "what style", "what type", "what kind"
    ],
    "pattern": [
        "what pattern", "patterned", "design"
    ],
    "shape": [
        "what shape"
    ],
    "brand": [
        "what brand", "what logo", "brand or logo"
    ],
    "position": [
        "where is", "what position", "located", "placement"
    ],
    "count": [
        "how many", "number of"
    ],
    "content": [
        "what is shown", "what is on", "what appears on"
    ],
    "decoration": [
        "decorative", "decorations", "features"
    ],
    "fabric": [
        "what fabric"
    ]
}

# test set name -> (questions file suffix in data_files/, question field)
TEST_SETS = {
    "adversary_presupposition": ("coco_{subset}_difficult_samples_adversary_questions", "adversary_question"),
    "real_recognition": ("{subset}_pope_questions", "pope_question"),
    "adversary_recognition": ("{subset}_pope_questions_negative", "pope_question"),
    "random_recognition": ("{subset}_pope_questions_random", "pope_question"),
}


def extract_question_intent(question: str) -> str:
    q = question.lower()

    for intent, patterns in QUESTION_INTENT_PATTERNS.items():
        for p in patterns:
            if p in q:
                return intent

    return "other"


def k_center_greedy(embeddings, K):
    n = embeddings.shape[0]
    selected = []
    random.seed(42)
    seed = random.randint(0, n - 1)
    selected.append(seed)

    distances = cosine_distances(embeddings[selected], embeddings).flatten()
    distances[seed] = -np.inf

    for _ in tqdm(range(K - 1), desc="K-Center Greedy selection"):
        idx = np.argmax(distances)
        selected.append(idx)
        new_distances = cosine_distances(
            embeddings[[idx]],
            embeddings
        ).flatten()

        distances = np.minimum(distances, new_distances)
        distances[idx] = -np.inf

    return selected


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", type=str, nargs="+", default=["florence", "instances"])

    # Relative paths below are resolved against the project root
    parser.add_argument("--save_dir", type=str, default="data_files")
    parser.add_argument("--output_dir", type=str, default="data_files/test_dataset")

    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--intent_weight", type=float, default=2.0)
    parser.add_argument("--object_weight", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=3000, help="Images selected per subset.")
    args = parser.parse_args()

    save_dir = PROJECT_ROOT / args.save_dir  # an absolute path overrides PROJECT_ROOT
    output_dir = PROJECT_ROOT / args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model = SentenceTransformer(args.embedding_model, device="cuda" if torch.cuda.is_available() else "cpu")

    for subset in args.subsets:
        samples = load_jsonl(save_dir / f"coco_{subset}_adversary_random_present_objects_filtered.json")
        image_to_objects = {s["img_filename"]: (s["present_object"], s["absent_object"], s["random_object"]) for s in samples}

        # Select over the adversary questions: their intent and their target (absent) object
        adversary_questions = load_jsonl(save_dir / f"coco_{subset}_difficult_samples_adversary_questions.json")
        intent_phrases = [extract_question_intent(s["adversary_question"]) for s in adversary_questions]
        object_phrases = [s["absent_object"] for s in adversary_questions]

        print("Embedding question intents...")
        intent_emb = model.encode(intent_phrases, batch_size=args.batch_size, show_progress_bar=True).astype(np.float32)
        print("Embedding target objects...")
        object_emb = model.encode(object_phrases, batch_size=args.batch_size, show_progress_bar=True).astype(np.float32)

        combined_embeddings = np.concatenate([args.intent_weight * intent_emb, args.object_weight * object_emb], axis=1)
        print("Combined embedding shape:", combined_embeddings.shape)

        print("Running K-Center Greedy selection...")
        selected_indices = k_center_greedy(combined_embeddings, args.num_samples)
        selected_images = {adversary_questions[idx]["img_filename"] for idx in selected_indices}

        # The same selected images in every test set
        for test_set, (questions_file, question_field) in TEST_SETS.items():
            questions = load_jsonl(save_dir / f"{questions_file.format(subset=subset)}.json")
            items = [
                {
                    "image": q["img_filename"],
                    "present_object": image_to_objects[q["img_filename"]][0],
                    "absent_object": image_to_objects[q["img_filename"]][1],
                    "random_object": image_to_objects[q["img_filename"]][2],
                    "text": q[question_field],
                }
                for q in questions if q["img_filename"] in selected_images
            ]
            save_path = output_dir / f"{test_set}_{subset}_balanced_w_category.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
            print(f"{len(items)} {test_set} questions -> {save_path}")
