# Adapted from https://github.com/Wu0409/Antidote/blob/main/data_pipeline/2_1_image_generate.py
import argparse
import multiprocessing
import os

import torch
from diffusers import StableDiffusion3Pipeline

from utils import read_jsonl

NEGATIVE_SUFFIX = (", low-quality, over-smooth, over-saturated, bad anatomy, disconnected limbs, extra limbs, "
                   "extra fingers, mutation, deformed, duplicate")


def generate_range(start, end, gpu_id, save_dir, triplets):
    torch.cuda.set_device(gpu_id)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
    ).to("cuda")

    for idx in range(start, end):
        save_path = os.path.join(save_dir, f"{idx}.jpg")
        if os.path.exists(save_path):
            continue
        image = pipe(
            prompt=triplets[idx]["prompt"],
            negative_prompt=triplets[idx]["prompt_neg"] + NEGATIVE_SUFFIX,
            num_inference_steps=28,
            guidance_scale=7.5,
        ).images[0]
        image.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate one SD3 image per triplet (gated model: set HF_TOKEN).")
    parser.add_argument("--triplets", required=True, help="Output of 1_triplets.py.")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--num_gpus", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    triplets = read_jsonl(args.triplets)
    per_gpu = len(triplets) // args.num_gpus

    multiprocessing.set_start_method("spawn")
    processes = []
    for gpu_id in range(args.num_gpus):
        start = gpu_id * per_gpu
        end = len(triplets) if gpu_id == args.num_gpus - 1 else (gpu_id + 1) * per_gpu
        p = multiprocessing.Process(target=generate_range, args=(start, end, gpu_id, args.save_dir, triplets))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
