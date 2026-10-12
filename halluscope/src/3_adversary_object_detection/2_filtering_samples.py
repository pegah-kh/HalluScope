"""
Only keeping those samples where the image is associated with more than `min_objects` objects,
since we want to capture the co-occurrence of multiple objects in the same image, and only want to keep images where the context is rich.
"""
import os
import json
import argparse
from pathlib import Path


# halluscope/halluscope/, two levels above this file's folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Relative paths below are resolved against the project root
    parser.add_argument("--objects_file", type=str, default="data_files/florence_dino_filtered_objects.json", help="Output of 3_1 or 3_2.")
    parser.add_argument("--objects_name", type=str, default="coco_florence_objects", help="coco_florence_objects (3_1) or coco_instances_objects (3_2).")
    parser.add_argument("--min_objects", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="data_files")
    parser.add_argument("--unfiltered_filename", type=str, default=None, help="Default: output of 1_context_aware_adversary_discovery.py for --objects_name")
    parser.add_argument("--save_filename", type=str, default=None, help="Default: --unfiltered_filename + _filtered")
    args = parser.parse_args()

    save_dir = PROJECT_ROOT / args.save_dir
    unfiltered_filename = args.unfiltered_filename or args.objects_name.removesuffix("_objects") + "_adversary_random_present_objects"
    save_filename = args.save_filename or unfiltered_filename + "_filtered"

    with open(PROJECT_ROOT / args.objects_file, "r", encoding="utf-8") as f:
        objects_per_sample = json.load(f)[args.objects_name]

    with open(os.path.join(save_dir, unfiltered_filename + ".json"), "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    save_path = os.path.join(save_dir, save_filename + ".json")
    kept = 0
    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            objects = objects_per_sample[sample["img_id"]]
            if len(objects) > args.min_objects:
                sample["present"] = objects
                f.write(json.dumps(sample) + "\n")
                kept += 1

    print(f"Kept {kept} out of {len(samples)} samples ({save_path})")
