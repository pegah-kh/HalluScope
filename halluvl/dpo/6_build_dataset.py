import argparse
import os
import random

from utils import read_json, write_json

TYPES = ["cpq", "tpq", "pope", "pope_negative", "description_unilateral", "description_injection",
         "description_contrastive", "attribute"]
PROPORTIONS = [5000, 5000, 3000, 3000, 3000, 3000, 3000, 5000]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the full dataset and the training mixture from the per-type pair pools.")
    parser.add_argument("--pairs_dir", required=True, help="Folder with one <type>.json pool per pair type.")
    parser.add_argument("--model", choices=["qwen", "llava"], required=True)
    parser.add_argument("--total_samples", type=int, default=20000)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    scale = args.total_samples / sum(PROPORTIONS)

    full, mixture = [], []
    for pair_type, proportion in zip(TYPES, PROPORTIONS):
        pool = [{"id": p["id"], "image": p["image"], "type": pair_type, "prompt": p["prompt"],
                 "chosen": p["chosen"], "rejected": p["rejected"]}
                for p in read_json(os.path.join(args.pairs_dir, pair_type + ".json"))]
        share = int(proportion * scale)
        sampled = pool if share >= len(pool) else random.sample(pool, share)
        print(f"{pair_type}: {len(sampled)} of {len(pool)} pairs (share {share})")
        full.extend(pool)
        mixture.extend(sampled)
    random.shuffle(mixture)

    os.makedirs(args.output_dir, exist_ok=True)
    write_json(full, os.path.join(args.output_dir, f"{args.model}_full.json"))
    write_json(mixture, os.path.join(args.output_dir, f"{args.model}_mixture.json"))
    print(f"{args.model}_full: {len(full)} pairs, {args.model}_mixture: {len(mixture)} pairs")
