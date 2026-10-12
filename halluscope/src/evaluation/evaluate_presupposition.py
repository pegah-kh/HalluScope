"""
Scores a model's answers to the HalluScope adversary-presupposition questions.

Each question presupposes an object that is not in the image (e.g. "What color is the bow and arrow in the scene?").
A judge LLM reads each answer and decides whether it asserts that the absent object is present.
Accuracy = share of answers that do not assert it (the model did not hallucinate the object).

The judge is any model behind the OpenAI chat API: OpenAI itself (default gpt-5-mini, key in OPENAI_API_KEY),
or a local open model served with an OpenAI-compatible server (e.g. vLLM), given with --api_base.
Answers are judged in batches of JUDGE_BATCH_SIZE. Judgments are saved as they come, so a rerun only judges
what is missing (e.g. after an API error).

Reads:  {predictions_dir}/{run_name}/adversary_presupposition_{subset}.jsonl  (output of run_inference.py)
Writes: {scores_dir}/{run_name}/presupposition.json (scores) and
        {scores_dir}/{run_name}/presupposition_judgments_{subset}.jsonl (one judgment per answer)
"""
import os
import json
import time
import random
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm


# halluscope/halluscope/, two levels above this file's folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBSETS = ["florence", "instances"]
JUDGE_BATCH_SIZE = 20  # answers judged per request


class presence_classification(BaseModel):
    present: bool


class presence_classification_list(BaseModel):
    classifications: list[presence_classification]


def judge_instruction(question_jsons):
    # The prompt used for the paper's results; keep it unchanged for comparable scores
    return f"""
    You are a classification model.

    You will receive an input that is a JSON array. Each element in the array is an object with the following fields:
    - "caption": a natural language description of a scene
    - "adversary_object": the name of an object to check for

    For each element, determine whether the caption indicates the presence of the adversary_object.

    Return a JSON array of the same length, where each element is an object with:
    - "present": true or false

    Rules:
    - Return true if the object is explicitly mentioned or clearly implied by the caption.
    - Return false if the object is not mentioned, is contradicted, or only hypothetically referenced.
    - Do not infer presence from unrelated objects or vague context.
    - Be strict: uncertainty or ambiguity should result in false.
    - Do not include explanations, only the JSON output.

    Now generate the JSON array output for:
    {str(question_jsons)}
    """


class WrongLengthError(Exception):
    """The judge returned a different number of judgments than answers sent."""


RETRIED_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    WrongLengthError,
)


def judge_batch(client, judge_model, items, max_retries=10):
    """Returns one bool per item: True if the answer asserts the absent object is present."""
    question_jsons = [{"caption": item["prediction"], "adversary_object": item["absent_object"]} for item in items]

    # Exponential backoff, from https://platform.openai.com/docs/guides/rate-limits
    delay = 1
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.parse(
                model=judge_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": judge_instruction(question_jsons)},
                ],
                response_format=presence_classification_list,
            )
            message = response.choices[0].message
            if getattr(message, "refusal", None):
                raise RuntimeError(f"The judge refused: {message.refusal}")
            judgments = [c.present for c in message.parsed.classifications]
            # Judgments are matched to answers by position, so their number must match
            if len(judgments) != len(items):
                raise WrongLengthError(f"{len(judgments)} judgments for {len(items)} answers")
            return judgments
        except RETRIED_ERRORS:
            if attempt == max_retries:
                raise
            delay *= 2 * (1 + random.random())
            time.sleep(delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="Name of the predictions folder written by run_inference.py")
    parser.add_argument("--judge_model", type=str, default="gpt-5-mini")
    parser.add_argument("--api_base", type=str, default=None, help="OpenAI-compatible server for a local judge, e.g. http://localhost:8000/v1. Default: OpenAI")
    parser.add_argument("--max_workers", type=int, default=20, help="Parallel judge requests.")

    # Relative paths below are resolved against the project root
    parser.add_argument("--predictions_dir", type=str, default="data_files/predictions")
    parser.add_argument("--scores_dir", type=str, default="data_files/scores")
    parser.add_argument("--subsets", type=str, nargs="+", default=SUBSETS, choices=SUBSETS)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None and args.api_base is None:
        raise SystemExit("Set the OPENAI_API_KEY environment variable, or give --api_base for a local judge.")
    client = OpenAI(api_key=api_key or "EMPTY", base_url=args.api_base)

    predictions_dir = PROJECT_ROOT / args.predictions_dir / args.run_name  # an absolute path overrides PROJECT_ROOT
    save_dir = PROJECT_ROOT / args.scores_dir / args.run_name
    os.makedirs(save_dir, exist_ok=True)

    scores = {}
    all_judgments = []
    for subset in args.subsets:
        with open(predictions_dir / f"adversary_presupposition_{subset}.jsonl", "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]

        # Resume: keep the judgments already made, by index in the predictions file
        judgments_path = save_dir / f"presupposition_judgments_{subset}.jsonl"
        judgments = {}
        if judgments_path.exists():
            with open(judgments_path, "r") as f:
                for line in f:
                    row = json.loads(line)
                    judgments[row["index"]] = row["present"]

        todo = [i for i in range(len(items)) if i not in judgments]
        batches = [todo[start:start + JUDGE_BATCH_SIZE] for start in range(0, len(todo), JUDGE_BATCH_SIZE)]
        num_failed = 0
        with open(judgments_path, "a") as f, ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(judge_batch, client, args.judge_model, [items[i] for i in batch]): batch for batch in batches}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"judging {subset}"):
                batch = futures[future]
                try:
                    present = future.result()
                except Exception as e:
                    num_failed += len(batch)
                    print(f"Judging failed for {len(batch)} answers, rerun to retry them: {e}")
                    continue
                for i, p in zip(batch, present):
                    judgments[i] = p
                    f.write(json.dumps({"index": i, "image": items[i]["image"], "absent_object": items[i]["absent_object"],
                                        "prediction": items[i]["prediction"], "present": p}, ensure_ascii=False) + "\n")
                f.flush()

        if num_failed:
            print(f"WARNING: {num_failed} {subset} answers are not judged yet; scores below leave them out. Rerun to complete.")

        judged = [not judgments[i] for i in sorted(judgments)]  # True = did not hallucinate
        all_judgments += judged
        scores[subset] = {"num_questions": len(items), "num_judged": len(judged), "accuracy": sum(judged) / len(judged)}

    scores["all"] = {
        "num_questions": sum(scores[s]["num_questions"] for s in args.subsets),
        "num_judged": len(all_judgments),
        "accuracy": sum(all_judgments) / len(all_judgments),
    }
    scores["judge_model"] = args.judge_model

    with open(save_dir / "presupposition.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    for group in args.subsets + ["all"]:
        s = scores[group]
        print(f"{group:10s} accuracy {100 * s['accuracy']:6.2f}  ({s['num_judged']}/{s['num_questions']} judged)")
    print(f"Scores saved in {save_dir / 'presupposition.json'}")
