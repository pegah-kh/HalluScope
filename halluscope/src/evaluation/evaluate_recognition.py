"""
Scores a model's answers to the HalluScope recognition (yes/no) questions.

Three test sets, per subset (florence, instances):
- real_recognition:      about the present object, expected answer "yes"
- adversary_recognition: about the absent, context-related object, expected answer "no"
- random_recognition:    about the absent, random object, expected answer "no"

An answer is the first whole word "yes" or "no" in the prediction. Answers with neither (e.g. "I don't know")
are invalid: they count as wrong and are reported separately.

Metrics:
- per test set: accuracy, share of "yes" answers, number of invalid answers;
- POPE-style, on real + adversary ("adversarial") and real + random ("random"):
  accuracy, precision, recall and F1 of the "yes" class, and share of "yes" answers.
All metrics are given per subset and over both subsets together ("all").

Reads:  {predictions_dir}/{run_name}/{task}_{subset}.jsonl  (output of run_inference.py)
Writes: {scores_dir}/{run_name}/recognition.json
"""
import os
import re
import json
import argparse
from pathlib import Path


# halluscope/halluscope/, two levels above this file's folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_ANSWERS = {
    "real_recognition": "yes",
    "adversary_recognition": "no",
    "random_recognition": "no",
}
POPE_SETTINGS = {
    "adversarial": ["real_recognition", "adversary_recognition"],
    "random": ["real_recognition", "random_recognition"],
}
SUBSETS = ["florence", "instances"]


def parse_answer(prediction):
    """'yes', 'no', or None when the prediction contains neither word."""
    match = re.search(r"\b(yes|no)\b", prediction.lower())
    return match.group(1) if match else None


def test_set_scores(answers, expected):
    return {
        "num_questions": len(answers),
        "accuracy": sum(answer == expected for answer in answers) / len(answers),
        "yes_ratio": sum(answer == "yes" for answer in answers) / len(answers),
        "num_invalid": sum(answer is None for answer in answers),
    }


def pope_scores(answers_and_expected):
    """Scores of the "yes" class. An invalid answer is wrong: a false negative for a "yes" question, a false positive for a "no" one."""
    tp = sum(expected == "yes" and answer == "yes" for answer, expected in answers_and_expected)
    fn = sum(expected == "yes" and answer != "yes" for answer, expected in answers_and_expected)
    tn = sum(expected == "no" and answer == "no" for answer, expected in answers_and_expected)
    fp = sum(expected == "no" and answer != "no" for answer, expected in answers_and_expected)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "num_questions": len(answers_and_expected),
        "accuracy": (tp + tn) / len(answers_and_expected),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "yes_ratio": sum(answer == "yes" for answer, _ in answers_and_expected) / len(answers_and_expected),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="Name of the predictions folder written by run_inference.py")

    # Relative paths below are resolved against the project root
    parser.add_argument("--predictions_dir", type=str, default="data_files/predictions")
    parser.add_argument("--scores_dir", type=str, default="data_files/scores")
    parser.add_argument("--subsets", type=str, nargs="+", default=SUBSETS, choices=SUBSETS)
    args = parser.parse_args()

    predictions_dir = PROJECT_ROOT / args.predictions_dir / args.run_name  # an absolute path overrides PROJECT_ROOT

    # answers[task][subset] = parsed answers
    answers = {}
    for task in EXPECTED_ANSWERS:
        answers[task] = {}
        for subset in args.subsets:
            with open(predictions_dir / f"{task}_{subset}.jsonl", "r", encoding="utf-8") as f:
                answers[task][subset] = [parse_answer(json.loads(line)["prediction"]) for line in f if line.strip()]

    scores = {}
    for group in args.subsets + ["all"]:
        subsets = args.subsets if group == "all" else [group]
        scores[group] = {
            task: test_set_scores([a for s in subsets for a in answers[task][s]], expected)
            for task, expected in EXPECTED_ANSWERS.items()
        }
        scores[group]["pope"] = {
            setting: pope_scores([(a, EXPECTED_ANSWERS[task]) for task in tasks for s in subsets for a in answers[task][s]])
            for setting, tasks in POPE_SETTINGS.items()
        }

    save_dir = PROJECT_ROOT / args.scores_dir / args.run_name
    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir / "recognition.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print(f"{'':10s} {'test set':22s} {'accuracy':>9s} {'yes ratio':>10s} {'invalid':>8s}")
    for group in args.subsets + ["all"]:
        for task in EXPECTED_ANSWERS:
            s = scores[group][task]
            print(f"{group:10s} {task:22s} {100 * s['accuracy']:9.2f} {100 * s['yes_ratio']:10.2f} {s['num_invalid']:8d}")
    print(f"\n{'':10s} {'POPE setting':22s} {'accuracy':>9s} {'precision':>10s} {'recall':>8s} {'F1':>8s}")
    for group in args.subsets + ["all"]:
        for setting in POPE_SETTINGS:
            s = scores[group]["pope"][setting]
            print(f"{group:10s} {setting:22s} {100 * s['accuracy']:9.2f} {100 * s['precision']:10.2f} {100 * s['recall']:8.2f} {100 * s['f1']:8.2f}")
    print(f"\nScores saved in {save_dir / 'recognition.json'}")
