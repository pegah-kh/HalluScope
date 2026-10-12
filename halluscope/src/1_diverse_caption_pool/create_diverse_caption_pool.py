"""
This file finds the coco samples that are most diverse using K-Center Greedy selection.
It saves two files:
1. selected_coco_indices_30k.npy: A numpy array containing the indices of the selected captions in the original COCO captions file.
2. selected_coco_metadata.json: A JSON file containing the corresponding img_ids, ids, and captions for the selected indices.


"""


import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import random
import argparse
from pathlib import Path

# halluscope/halluscope/, two levels above this file's folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

COCO_DIR = "/data/common/coco"
OUTPUT_DIR = PROJECT_ROOT / "data_files"
OUTPUT_NAME = "selected_coco"
NUM_SAMPLES = 30000
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 1024


indices_to_img_ids = {}
indices_to_ids = {}
indices_to_captions = {}


def load_coco_captions(caption_file):
    global indices_to_img_ids, indices_to_captions, indices_to_ids

    with open(caption_file, "r") as f:
        data = json.load(f)
    
    captions = []
    img_ids = []
    ids = []
    for idx, ann in enumerate(data["annotations"]):
        # print(ann.keys()) # dict_keys(['image_id', 'id', 'caption'])
        captions.append(ann["caption"])
        img_ids.append(ann["image_id"])
        ids.append(ann["id"])

        indices_to_img_ids[idx] = ann["image_id"]
        indices_to_ids[idx] = ann["id"]
        indices_to_captions[idx] = ann["caption"]

    return captions, img_ids, ids

@torch.no_grad()
def k_center_greedy(embeddings, K, device="cuda"):
    # Normalize once so cosine distance is just 1 - dot product
    emb = torch.nn.functional.normalize(torch.from_numpy(embeddings).to(device), dim=1)
    n = emb.shape[0]

    # Keep selected indices on the GPU to avoid a host sync every iteration
    selected = torch.empty(K, dtype=torch.long, device=device)
    seed = random.randint(0, n - 1)
    selected[0] = seed

    distances = 1.0 - emb @ emb[seed]

    for i in tqdm(range(1, K), desc="K-Center Greedy selection"):
        idx = torch.argmax(distances)
        selected[i] = idx
        torch.minimum(distances, 1.0 - emb @ emb[idx], out=distances)

    return selected.cpu().tolist()



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Create a diverse caption pool using K-Center Greedy selection.")
    argparser.add_argument("--coco_dir", type=str, default=COCO_DIR, help="COCO root folder (must contain annotations/captions_train2017.json).")
    argparser.add_argument("--num_samples", type=int, default=NUM_SAMPLES, help="Number of diverse captions to select.")
    argparser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL, help="SentenceTransformer model to use for embedding captions.")
    argparser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Folder to save outputs in. Relative paths are resolved against the project root.")
    argparser.add_argument("--output_name", type=str, default=OUTPUT_NAME, help="Prefix of the output files.")
    argparser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible selection.")
    args = argparser.parse_args()

    caption_file = Path(args.coco_dir) / "annotations" / "captions_train2017.json"
    output_dir = PROJECT_ROOT / args.output_dir  # an absolute output_dir overrides PROJECT_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = str(output_dir / args.output_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    
    print("Loading captions...")
    captions, img_ids, ids = load_coco_captions(caption_file)
    print(f"Loaded {len(captions)} captions")
    print(len(list(set(captions)))) # 570281
    print(len(list(set(img_ids)))) # 118287
    print(len(list(set(ids)))) # 591753


    print("Embedding captions...")
    model = SentenceTransformer(args.embedding_model, device="cuda")
    model.half()
    embeddings = model.encode(captions, batch_size=BATCH_SIZE, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    print("Embeddings shape:", embeddings.shape)

    print("Running K-Center Greedy selection...")
    selected_indices = k_center_greedy(embeddings, args.num_samples)



    # Saving the indices of the selected captions
    np.save(output_file+"_indices_30k.npy", np.array(selected_indices))
    print(f"Saved {len(selected_indices)} selected indices to {output_file}")


    # Now, let's also save the corresponding img_ids, ids, and captions for the selected indices
    filtered_img_ids, filtered_ids, filtered_captions = [], [], []
    seen_img_ids = set()

    for filtered_index in selected_indices:
        current_img_id = indices_to_img_ids[filtered_index]
        if current_img_id not in seen_img_ids:
            seen_img_ids.add(current_img_id)
            filtered_img_ids.append(current_img_id)
            filtered_ids.append(indices_to_ids[filtered_index])
            filtered_captions.append(indices_to_captions[filtered_index])


    filtered_metadata = {
        "img_ids": filtered_img_ids,
        "ids": filtered_ids,
        "captions": filtered_captions,
    }

    with open(output_file+"_metadata.json", "w", encoding="utf-8") as f:
        json.dump(filtered_metadata, f, indent=2, ensure_ascii=False)





