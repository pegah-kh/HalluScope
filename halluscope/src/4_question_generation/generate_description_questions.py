"""
Attaches a randomly chosen open-ended "describe the image" prompt to each stage-3 sample.

Reads:  data_files/coco_{subset}_adversary_random_present_objects_filtered.json (stage 3)
Writes: data_files/{subset}_description_questions.json (JSON Lines): the stage-3 fields plus "desc_question"
"""
import json
import random
import argparse
from pathlib import Path


# halluscope/halluscope/, two levels above this file's folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DESCRIPTION_QUESTIONS = [
    'Please describe the image in detail.',
    'Provide a detailed description of the image?',
    'Please break down the image and describe its elements.',
    'Please explain what is shown in the image in detail.',
    'Could you offer a thorough description of the image?',
    'Please provide an in-depth description of the image.',
    'Can you describe what you see in the image?',
]

# Fields kept from the stage-3 samples
FIELDS = ["img_id", "img_filename", "captions", "present_object", "absent_object", "present"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="florence", choices=["florence", "instances"])
    parser.add_argument("--seed", type=int, default=42, help="Seed for the prompt choice.")

    # Relative paths below are resolved against the project root
    parser.add_argument("--save_dir", type=str, default="data_files")
    parser.add_argument("--input_filename", type=str, default=None, help="Default: data_files/coco_{subset}_adversary_random_present_objects_filtered.json")
    parser.add_argument("--save_filename", type=str, default=None, help="Default: {subset}_description_questions")
    args = parser.parse_args()

    save_dir = PROJECT_ROOT / args.save_dir  # an absolute path overrides PROJECT_ROOT
    input_path = PROJECT_ROOT / (args.input_filename or f"data_files/coco_{args.subset}_adversary_random_present_objects_filtered.json")
    save_path = save_dir / f"{args.save_filename or args.subset + '_description_questions'}.json"

    with open(input_path, "r", encoding="utf-8") as f:
        items = [{field: json.loads(line)[field] for field in FIELDS} for line in f if line.strip()]

    rng = random.Random(args.seed)
    with open(save_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps({**item, "desc_question": rng.choice(DESCRIPTION_QUESTIONS)}, ensure_ascii=False) + "\n")
    print(f"{len(items)} questions -> {save_path}")
