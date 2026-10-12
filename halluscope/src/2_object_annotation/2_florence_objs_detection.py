"""
This code is for generating object detection annotations for a filtered set of COCO images using the Florence model.
These detected objects would later be filtered to check that detected objects are actually present in the image.

Output: data_files/coco_florence_objects.json = {"coco_florence_objects": {img_id: [object labels]}}
Images where Florence detects nothing are left out.

Images are split across all GPUs (accelerate processes x --num_shards jobs). Each GPU writes its own
part file, which can be resumed. With a single GPU the final file is written directly; otherwise run
--merge --num_parts <total GPUs> once every part is done.
"""
import os
import glob
import json
import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from transformers import AutoProcessor, AutoModelForCausalLM


# halluscope/halluscope/, two levels above this file's folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

COCO_DIR = "/data/common/coco"
CACHE_DIR = "/data/khayatan/cache"
FLORENCE_MODEL = "microsoft/Florence-2-large"
TASK_PROMPT = "<OD>"


class ImageDataset(Dataset):
    def __init__(self, images, image_dir):
        self.images = images
        self.image_dir = image_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image = Image.open(os.path.join(self.image_dir, img_info["file_name"])).convert("RGB")
        return img_info["id"], image


def collate(batch):
    img_ids, images = zip(*batch)
    return list(img_ids), list(images)


@torch.no_grad()
def detect_objects(model, processor, images, device, dtype):
    inputs = processor(text=[TASK_PROMPT] * len(images), images=images, return_tensors="pt").to(device, dtype)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False,
    )
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)

    all_objects = []
    for text, image in zip(generated_texts, images):
        parsed_answer = processor.post_process_generation(text, task=TASK_PROMPT, image_size=(image.width, image.height))
        all_objects.append(sorted(set(parsed_answer[TASK_PROMPT]["labels"])))
    return all_objects


def part_path(save_dir, save_filename, rank, world_size):
    return os.path.join(save_dir, f"{save_filename}_part{rank:03d}of{world_size:03d}.jsonl")


def merge_parts(save_dir, save_filename, num_parts):
    paths = sorted(glob.glob(part_path(save_dir, save_filename, 0, num_parts).replace("part000", "part*")))
    if len(paths) != num_parts:
        raise FileNotFoundError(f"Expected {num_parts} part files in {save_dir}, found {len(paths)}")
    # Each GPU writes a .done marker once its whole slice is processed, so a crashed part can't be merged
    unfinished = [path for path in paths if not os.path.exists(path + ".done")]
    if unfinished:
        raise RuntimeError(f"These parts did not finish (no .done marker), rerun their jobs first: {unfinished}")

    image_to_objects = {}
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                row = json.loads(line)
                if row["objects"]:
                    image_to_objects[row["id"]] = row["objects"]

    output_path = os.path.join(save_dir, save_filename + ".json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"coco_florence_objects": image_to_objects}, f, indent=2, ensure_ascii=False)
    print(f"Merged {len(paths)} parts -> {output_path} ({len(image_to_objects)} images with objects)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_dir", type=str, default=COCO_DIR, help="COCO root folder (must contain annotations/captions_train2017.json and train2017/).")
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR, help="Hugging Face cache folder holding the Florence model.")

    # Relative paths below are resolved against the project root
    parser.add_argument("--filtered_metadata_path", type=str, default="data_files/selected_coco_metadata.json", help="Output of step 1_diverse_caption_pool.")
    parser.add_argument("--save_dir", type=str, default="data_files")
    parser.add_argument("--save_filename", type=str, default="coco_florence_objects")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    # Multi-node: launch num_shards jobs (e.g. a SLURM array), each with its own shard_id.
    # Each job is further split across its own GPUs by accelerate.
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)

    # Merge mode: combine the part files of a finished multi-GPU run into the final JSON
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--num_parts", type=int, default=None, help="With --merge: total number of GPUs the run used.")
    args = parser.parse_args()

    save_dir = PROJECT_ROOT / args.save_dir

    if args.merge:
        merge_parts(save_dir, args.save_filename, args.num_parts)
        raise SystemExit

    annotation_file = os.path.join(args.coco_dir, "annotations", "captions_train2017.json")
    image_dir = os.path.join(args.coco_dir, "train2017")
    filtered_metadata_path = PROJECT_ROOT / args.filtered_metadata_path  # an absolute path overrides PROJECT_ROOT

    accelerator = Accelerator()
    device = accelerator.device
    dtype = torch.float16

    # Each GPU across all jobs gets a disjoint slice of the images
    rank = args.shard_id * accelerator.num_processes + accelerator.process_index
    world_size = args.num_shards * accelerator.num_processes

    os.makedirs(save_dir, exist_ok=True)
    save_path = part_path(save_dir, args.save_filename, rank, world_size)

    # Resume: skip images already processed in this GPU's part file
    done_ids = set()
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            done_ids = {json.loads(line)["id"] for line in f if line.strip()}

    with open(filtered_metadata_path, "r", encoding="utf-8") as f:
        filtered_img_ids = set(json.load(f)["img_ids"])

    with open(annotation_file, "r") as f:
        images = [img for img in json.load(f)["images"] if img["id"] in filtered_img_ids]

    images = [img for img in images[rank::world_size] if img["id"] not in done_ids]
    print(f"[rank {rank}/{world_size}] {len(images)} images to process ({len(done_ids)} already done) -> {save_path}")

    loader = DataLoader(ImageDataset(images, image_dir), batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate)

    model = AutoModelForCausalLM.from_pretrained(FLORENCE_MODEL, torch_dtype=dtype, trust_remote_code=True, cache_dir=args.cache_dir).to(device).eval()
    processor = AutoProcessor.from_pretrained(FLORENCE_MODEL, trust_remote_code=True, cache_dir=args.cache_dir)

    with open(save_path, "a") as f:
        for img_ids, batch_images in tqdm(loader, desc=f"rank {rank}"):
            for img_id, objects in zip(img_ids, detect_objects(model, processor, batch_images, device, dtype)):
                f.write(json.dumps({"id": img_id, "objects": objects}) + "\n")
            f.flush()  # so a killed job keeps what it finished

    open(save_path + ".done", "w").close()

    if world_size == 1:
        merge_parts(save_dir, args.save_filename, num_parts=1)
