"""
This code is for generating captions for a filtered set of COCO images using the Qwen model.
It loads a filtered set of COCO images, passes them through the Qwen model to generate captions, and saves the results in a JSON file.

"""
import os
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

from models import get_model_class


# halluscope/halluscope/, two levels above this file's folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

COCO_DIR = "/data/common/coco"
CACHE_DIR = "/data/khayatan/cache"



def ask_qwen_batched(qwen_model_class, image_paths, accelerator=None):

    batch_size = len(image_paths)

    # The caption is later prepended to "Is there a/an {object} in this image?" questions,
    # so it should list what is actually visible, without guesses or formatting
    instruction = (
        "Describe this image in a short plain-text paragraph of 2 to 4 sentences. "
        "Mention every clearly visible object, person and animal, "
        "and briefly say where they are relative to each other. "
        "Only describe what is actually visible: do not guess, interpret, or mention things that are absent. "
        "Do not use lists, headings or markdown."
    )
    instructions = [instruction] * batch_size
    responses = [""] * batch_size

    inputs = qwen_model_class.preprocessor(
        instruction=instructions,
        image_file=image_paths,
        response=responses,
        generation_mode=True,
        continue_final_message=False
    )

    if accelerator is not None:
        model_device = next(qwen_model_class.model_.parameters()).device

        inputs = {
            k: v.to(model_device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

    out = qwen_model_class.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
    )

    input_ids = inputs["input_ids"]
    if input_ids.ndim == 2:
        input_lengths = [seq.size(0) for seq in input_ids]  # variable length
    else:
        input_lengths = [input_ids.size(0)] * batch_size

    preds = []
    for i in range(batch_size):
        pred_ids = out[i, input_lengths[i]:]
        decoded = qwen_model_class.get_tokenizer().decode(
            pred_ids, skip_special_tokens=True
        )
        preds.append(decoded)

    return preds



class FilteredCocoDataset(Dataset):
    def __init__(self, filtered_metadata_path, annotation_file, image_dir):
        with open(filtered_metadata_path, "r", encoding="utf-8") as f:
            filtered_metadata = json.load(f)
        self.filtered_img_ids = set(filtered_metadata["img_ids"])

        with open(annotation_file, "r") as f:
            coco_data = json.load(f)

        images = coco_data["images"]

        self.images = [img for img in images if img["id"] in self.filtered_img_ids]
        self.image_dir = image_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_filename = img_info["file_name"]

        image_path = os.path.join(self.image_dir, img_filename)


        return {"image_path": image_path, "img_id": img_id, "file_name": img_filename}



def clean(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean(x) for x in obj]
    else:
        return obj
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_files_only", type=bool, default=True)
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR, help="Hugging Face cache folder holding the Qwen model.")

    parser.add_argument("--coco_dir", type=str, default=COCO_DIR, help="COCO root folder (must contain annotations/captions_train2017.json and train2017/).")

    # Relative paths below are resolved against the project root
    parser.add_argument("--filtered_metadata_path", type=str, default="data_files/selected_coco_metadata.json", help="Output of step 1_diverse_caption_pool.")
    parser.add_argument("--save_dir", type=str, default="data_files")
    parser.add_argument("--save_filename", type=str, default="qwen_captions")

    # Multi-node: launch num_shards jobs (e.g. a SLURM array), each with its own shard_id.
    # Each job is further split across its own GPUs by accelerate.
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)


    args = parser.parse_args()

    annotation_file = os.path.join(args.coco_dir, "annotations", "captions_train2017.json")
    image_dir = os.path.join(args.coco_dir, "train2017")
    filtered_metadata_path = PROJECT_ROOT / args.filtered_metadata_path  # an absolute path overrides PROJECT_ROOT
    save_dir = PROJECT_ROOT / args.save_dir

    def setup_seeds(seed=0):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        cudnn.benchmark = False
        cudnn.deterministic = True

    setup_seeds()

    accelerator = Accelerator()


    qwen_model_class = get_model_class(
        model_name_or_path="Qwen/Qwen3.5-9B",
        model_base="",
        processor_name=None,
        device=accelerator.device,
        logger=None,
        args=args,
    )

    # Batched generation needs left padding so every prompt ends right before the generated tokens
    qwen_model_class.get_tokenizer().padding_side = "left"

    # Each GPU across all jobs gets a disjoint slice of the images
    rank = args.shard_id * accelerator.num_processes + accelerator.process_index
    world_size = args.num_shards * accelerator.num_processes

    os.makedirs(save_dir, exist_ok=True)
    if world_size > 1:
        # One file per GPU; merge afterwards with: cat qwen_captions_part*.json > qwen_captions.json
        save_path = os.path.join(save_dir, f"{args.save_filename}_part{rank:03d}of{world_size:03d}.json")
    else:
        save_path = os.path.join(save_dir, args.save_filename+".json")

    # Resume: skip images already captioned in this GPU's output file
    done_ids = set()
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            done_ids = {json.loads(line)["id"] for line in f if line.strip()}

    dataset = FilteredCocoDataset(filtered_metadata_path, annotation_file, image_dir)
    dataset.images = [img for img in dataset.images[rank::world_size] if img["id"] not in done_ids]
    print(f"[rank {rank}/{world_size}] {len(dataset)} images to caption ({len(done_ids)} already done) -> {save_path}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = qwen_model_class.get_model()
    model = accelerator.prepare(model)

    with open(save_path, "a") as f:

        for batch in tqdm(loader):
            image_path = batch["image_path"]
            img_id = batch["img_id"]

            model_predictions = ask_qwen_batched(qwen_model_class, image_path, accelerator)
            
            for item_id, item_prediction, item_image in zip(img_id, model_predictions, batch['file_name']):
                sample_data = {
                    "id": item_id,
                    "prediction": item_prediction,
                    "image": item_image
                }
                f.write(json.dumps(clean(sample_data)) + "\n")
            f.flush()  # so a killed job keeps what it finished
