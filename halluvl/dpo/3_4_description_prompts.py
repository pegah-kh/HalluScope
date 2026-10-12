import argparse
import random
import re

from utils import read_jsonl, write_jsonl

DESCRIPTION_QUESTIONS = [
    "Please describe the image in detail.",
    "Provide a detailed description of the image?",
    "Please break down the image and describe its elements.",
    "Please explain what is shown in the image in detail.",
    "Could you offer a thorough description of the image?",
    "Please provide an in-depth description of the image.",
    "Can you describe what you see in the image?",
]


def strip_prompt_weights(caption):
    return re.sub(r"\([^\)]+:\d+(\.\d+)?\)", lambda m: m.group(0).split(":")[0][1:], caption)


def unilateral_prompt(item, question):
    """Plain question vs. question with a correct hint (caption, present and absent objects)."""
    hint = (f"Given the hint of the image: "
            f"[the image caption: {strip_prompt_weights(item['caption'])}, "
            f"the object(s) you can see: {item['present']}, "
            f"the object(s) you cannot see: {item['no_exist']}], ")
    return {"image": f"{item['idx']}.jpg", "text": question, "text_hint": hint + question.lower()}


def contrastive_prompt(item, question):
    """Correct hint (present objects) vs. wrong hint (absent objects claimed visible)."""
    question = question.lower()
    return {
        "image": f"{item['idx']}.jpg",
        "text": f"Given the hint of the image: the object(s) you can see: {item['present']}, {question}",
        "text_hint": f"Given the hint of the image: the object(s) you can see: {item['no_exist']}, {question}",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build description prompts for the unilateral and contrastive hint strategies.")
    parser.add_argument("--verified", required=True, help="Output of 2_3_filter_detections.py.")
    parser.add_argument("--output_unilateral", required=True)
    parser.add_argument("--output_contrastive", required=True)
    args = parser.parse_args()

    items = read_jsonl(args.verified)
    questions = [random.choice(DESCRIPTION_QUESTIONS) for _ in items]
    write_jsonl([unilateral_prompt(i, q) for i, q in zip(items, questions)], args.output_unilateral)
    write_jsonl([contrastive_prompt(i, q) for i, q in zip(items, questions)], args.output_contrastive)
