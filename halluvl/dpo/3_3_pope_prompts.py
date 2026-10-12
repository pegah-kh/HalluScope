import argparse
import random

from utils import add_article, read_jsonl, write_jsonl

QUESTION_TEMPLATES = [
    "Is there {obj} in the image?",
    "Do you see {obj} in the picture?",
    "Can you find {obj} in this image?",
    "Does the image contain {obj}?",
    "Is {obj} present in the image?",
    "Would you say there's {obj} visible?",
    "Could this image include {obj}?",
]


def pope_question(item, negative):
    obj = item["no_exist"][0] if negative else item["present"][0]
    return {"idx": item["idx"], "object": obj,
            "pope_question": random.choice(QUESTION_TEMPLATES).format(obj=add_article(obj))}


def to_prompt(question, negative):
    """The hint always contradicts the image: absent for a present object, present for an absent one."""
    if negative:
        hint = f"Hint: There is {add_article(question['object'])} in the image."
    else:
        hint = f"Hint: There is no {question['object']} in the image."
    return {"image": f"{question['idx']}.jpg", "text": question["pope_question"],
            "text_hint": f"{hint} {question['pope_question']}"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build plain and hinted POPE-style yes/no prompts for model inference.")
    parser.add_argument("--verified", required=True, help="Output of 2_3_filter_detections.py.")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--negative", action="store_true", help="Ask about the first absent object instead of the first present one.")
    args = parser.parse_args()

    questions = [pope_question(item, args.negative) for item in read_jsonl(args.verified)]
    write_jsonl([to_prompt(q, args.negative) for q in questions], args.output_path)
