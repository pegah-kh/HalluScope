import argparse
import random

from utils import read_jsonl, write_jsonl


def absence_hint(obj):
    """Hint claiming `obj` is absent: correct for the CPQ object, false for the TPQ object."""
    return random.choice([
        f"There is no {obj} in the image. Answer naturally and explain why the question doesn’t apply.",
        f"Since no {obj} is shown, respond briefly and clearly that it’s not relevant to the image.",
        f"The image lacks a {obj}. Give a short, natural reply that reflects that.",
        f"No {obj} appears here.",
    ])


def to_prompt(question, kind):
    text = question[f"{kind}_question"].lower()
    return {
        "image": f"{question['idx']}.jpg",
        "text": text,
        "hall_object": question["hall_object"].lower(),
        "truth_object": question["truth_object"].lower(),
        "text_hint": absence_hint(question[f"{kind}_object"]) + " " + text,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build plain and hinted CPQ/TPQ prompts for model inference.")
    parser.add_argument("--questions", required=True, help="Output of 3_1_cpq_tpq_questions.py.")
    parser.add_argument("--output_tpq", required=True)
    parser.add_argument("--output_cpq", required=True)
    args = parser.parse_args()

    random.seed(0)
    tpq, cpq = [], []
    for question in read_jsonl(args.questions):
        tpq.append(to_prompt(question, "truth"))
        cpq.append(to_prompt(question, "hall"))

    write_jsonl(tpq, args.output_tpq)
    write_jsonl(cpq, args.output_cpq)
