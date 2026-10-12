"""
Generates three POPE-style yes/no questions per stage-3 sample, each from a randomly chosen template:
- about the present object (expected answer "yes")                 -> {subset}_pope_questions.json
- about the absent, context-related object (expected answer "no")  -> {subset}_pope_questions_negative.json
- about the absent, random object (expected answer "no")           -> {subset}_pope_questions_random.json

Reads:  data_files/coco_{subset}_adversary_random_present_objects_filtered.json (stage 3)
Writes: the three files above in data_files/ (JSON Lines): the stage-3 fields plus "pope_question"
"""
import json
import random
import argparse
from pathlib import Path


# halluscope/halluscope/, two levels above this file's folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# (singular, plural) forms of each template
QUESTION_TEMPLATES = [
    ("Is there {obj} in the image?", "Are there {obj} in the image?"),
    ("Do you see {obj} in the picture?", "Do you see {obj} in the picture?"),
    ("Can you find {obj} in this image?", "Can you find {obj} in this image?"),
    ("Does the image contain {obj}?", "Does the image contain {obj}?"),
    ("Is {obj} present in the image?", "Are {obj} present in the image?"),
    ("Would you say there's {obj} visible?", "Would you say there are {obj} visible?"),
    ("Could this image include {obj}?", "Could this image include {obj}?"),
]

# Which object each question set asks about, and the suffix of its file
QUESTION_SETS = {
    "": "present_object",
    "_negative": "absent_object",
    "_random": "random_object",
}

# Fields kept from the stage-3 samples
FIELDS = ["img_id", "img_filename", "captions", "present_object", "absent_object", "random_object", "present"]


def is_plural(phrase: str) -> bool:
    # crude plural check
    phrase = phrase.strip()
    return phrase.endswith('s') and not phrase.endswith('ss') and phrase != "bus"


def add_article(phrase: str) -> str:
    """Automatically add 'a' or 'an' where natural; leave plural or mass nouns unchanged."""
    phrase = phrase.strip()
    first_word = phrase.split()[0].lower()
    vowels = ('a', 'e', 'i', 'o', 'u')
    if is_plural(phrase):
        return phrase  # plural form, no article
    if first_word[0] in vowels:
        return f"an {phrase}"
    return f"a {phrase}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="florence", choices=["florence", "instances"])
    parser.add_argument("--seed", type=int, default=42, help="Seed for the template choice.")

    # Relative paths below are resolved against the project root
    parser.add_argument("--save_dir", type=str, default="data_files")
    parser.add_argument("--input_filename", type=str, default=None, help="Default: data_files/coco_{subset}_adversary_random_present_objects_filtered.json")
    parser.add_argument("--save_filename", type=str, default=None, help="Default: {subset}_pope_questions")
    args = parser.parse_args()

    save_dir = PROJECT_ROOT / args.save_dir  # an absolute path overrides PROJECT_ROOT
    input_path = PROJECT_ROOT / (args.input_filename or f"data_files/coco_{args.subset}_adversary_random_present_objects_filtered.json")
    save_filename = args.save_filename or f"{args.subset}_pope_questions"

    with open(input_path, "r", encoding="utf-8") as f:
        items = [{field: json.loads(line)[field] for field in FIELDS} for line in f if line.strip()]

    rng = random.Random(args.seed)
    for suffix, object_field in QUESTION_SETS.items():
        with open(save_dir / f"{save_filename}{suffix}.json", "w", encoding="utf-8") as f:
            for item in items:
                singular, plural = rng.choice(QUESTION_TEMPLATES)
                obj = item[object_field]
                question = (plural if is_plural(obj) else singular).format(obj=add_article(obj))
                f.write(json.dumps({**item, "pope_question": question}, ensure_ascii=False) + "\n")
        print(f"{len(items)} questions -> {save_dir / (save_filename + suffix + '.json')}")
