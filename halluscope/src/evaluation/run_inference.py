"""
Runs a Hugging Face vision-language model on the HalluScope test sets and saves its answers.

Works with chat VLMs that load with AutoModelForImageTextToText and whose processor has a chat template
(e.g. Qwen2.5-VL, Qwen3.5, LLaVA-1.5-hf, InternVL-hf). Answers are generated greedily.

Reads:  benchmark/{version}/{task}_{subset}.json (v1: the paper's test set, v2: regenerated with
        this code), or --test_dir (e.g. data_files/test_dataset, the output of src/4_question_generation)
Writes: {output_dir}/{run_name}/{task}_{subset}.jsonl, one line per question: the test item plus
        "task", "subset" and "prediction" (the model's answer). This is the input of the evaluation scripts.
A partially written file is resumed where it stopped.
"""
import os
import json
import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText


# halluscope/halluscope/, two levels above this file's folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TASKS = ["adversary_presupposition", "real_recognition", "adversary_recognition", "random_recognition"]
SUBSETS = ["florence", "instances"]


def test_file(test_dir, task, subset):
    return os.path.join(test_dir, f"{task}_{subset}.json")


@torch.no_grad()
def answer_batch(model, processor, images, questions, max_new_tokens):
    prompts = [
        processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,  # for models that think by default (e.g. Qwen3.5); ignored by the others
        )
        for question in questions
    ]
    inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(model.device, model.dtype)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[1]  # same for every row with left padding
    return [answer.strip() for answer in processor.batch_decode(out[:, input_len:], skip_special_tokens=True)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model id or local path, e.g. Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--run_name", type=str, default=None, help="Name of the predictions folder. Default: {model name}_{version}")
    parser.add_argument("--cache_dir", type=str, default=None, help="Hugging Face cache folder.")
    parser.add_argument("--coco_dir", type=str, required=True, help="COCO root folder (must contain train2017/).")

    # Relative paths below are resolved against the project root
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2"], help="Benchmark version in benchmark/: v1 (paper) or v2 (regenerated).")
    parser.add_argument("--test_dir", type=str, default=None, help="Use these test sets instead of --version, e.g. data_files/test_dataset.")
    parser.add_argument("--output_dir", type=str, default="data_files/predictions")
    parser.add_argument("--tasks", type=str, nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--subsets", type=str, nargs="+", default=SUBSETS, choices=SUBSETS)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="Only answer the first N questions of each test set, for a quick check.")
    args = parser.parse_args()

    run_name = args.run_name or f"{os.path.basename(args.model.rstrip('/'))}_{args.version}"
    test_dir = PROJECT_ROOT / (args.test_dir or f"benchmark/{args.version}")
    output_dir = PROJECT_ROOT / args.output_dir / run_name  # an absolute output_dir overrides PROJECT_ROOT
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(args.coco_dir, "train2017")

    processor = AutoProcessor.from_pretrained(args.model, cache_dir=args.cache_dir)
    # Batched generation needs left padding so every prompt ends right before the generated tokens
    processor.tokenizer.padding_side = "left"
    # device_map loads the weights straight onto the GPU, without a full copy in CPU memory
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda", cache_dir=args.cache_dir).eval()

    for task in args.tasks:
        for subset in args.subsets:
            with open(test_file(test_dir, task, subset), "r", encoding="utf-8") as f:
                items = json.load(f)[:args.limit]

            # Resume: skip the questions already answered in this file
            save_path = os.path.join(output_dir, f"{task}_{subset}.jsonl")
            num_done = 0
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    num_done = sum(1 for line in f if line.strip())

            with open(save_path, "a", encoding="utf-8") as f:
                for start in tqdm(range(num_done, len(items), args.batch_size), desc=f"{task} {subset}"):
                    batch = items[start:start + args.batch_size]
                    images = [Image.open(os.path.join(image_dir, item["image"])).convert("RGB") for item in batch]
                    answers = answer_batch(model, processor, images, [item["text"] for item in batch], args.max_new_tokens)
                    for item, answer in zip(batch, answers):
                        f.write(json.dumps({**item, "task": task, "subset": subset, "prediction": answer}, ensure_ascii=False) + "\n")
                    f.flush()  # so a killed run keeps what it finished

    print(f"Predictions saved in {output_dir}")
