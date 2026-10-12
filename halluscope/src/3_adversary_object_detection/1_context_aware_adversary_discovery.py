"""
This code is for discovering context-aware adversarial objects in COCO images using Grounding DINO and Qwen.
The goal is to find objects that are not present in the image, but are contextually related to the image content.

For each image with present objects (output of 3_1 or 3_2):
1. Candidate absent objects are ranked by how strongly they co-occur (PMI, smoothed with --pmi_smoothing) with the
   image's present objects.
   Never candidates: the objects of this image in any --exclude_objects_files, kept or rejected
   (rejected objects are uncertain, so they cannot be used as absent objects either).
2. The absent object is the best-ranked candidate (among the top TOP_K) that is confirmed absent:
   - DINO is not confident (best box score <= DINO_CONFIDENT_THRESH), and
   - Qwen answers no to "{caption} Is there a/an {object} in this image?" on the full image, and
   - if DINO found a box, Qwen also answers no on that box crop.
   A "yes" anywhere means the object may be there, so it is skipped.
3. The random object is drawn (with a fixed seed) from the other objects of the vocabulary, with the same
   exclusions and the same absence check.
4. The present object is the image's present object most related (PMI) to the absent object.
Images where no absent or no random object is confirmed are skipped. So are images with --min_objects present
objects or fewer, before any GPU work, since 2_filtering_samples.py drops them anyway (PMI still uses all images).

Output (one JSON per line): data_files/{coco_florence|coco_instances}_adversary_random_present_objects.json
with img_id, img_filename, captions (COCO), present_object, absent_object, random_object, all_present_objects.

Images are split across all GPUs (accelerate processes x --num_shards jobs). Each GPU writes its own
part file, which can be resumed. With a single GPU the final file is written directly; otherwise run
--merge --num_parts <total GPUs> once every part is done.
"""
import os
import sys
import glob
import json
import math
import random
import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# The Qwen wrapper lives in src/2_object_annotation/models
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2_object_annotation"))
from models import get_model_class


# halluscope/halluscope/, two levels above this file's folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

COCO_DIR = "/data/common/coco"
CACHE_DIR = "/data/khayatan/cache"
QWEN_MODEL = "Qwen/Qwen3.5-9B"
# Tiny (SwinT-OGC) was not trained on COCO, unlike the base (SwinB-cogcoor) checkpoint,
# so it judges these COCO train images independently
DINO_MODEL = "IDEA-Research/grounding-dino-tiny"

DINO_BOX_THRESH = 0.35
DINO_TEXT_THRESH = 0.4
DINO_CONFIDENT_THRESH = 0.6  # above this, DINO says the object is there
QWEN_MAX_NEW_TOKENS = 16  # only a yes/no answer is needed
MAX_BATCH = 8  # max objects per DINO / Qwen batch, bounds GPU memory

TOP_K = 10  # number of PMI-ranked candidates tried for the absent object
NUM_RANDOM_CANDIDATES = 3  # number of random candidates tried for the random object


def is_plural(phrase: str) -> bool:
    # crude plural check
    phrase = phrase.strip()
    return phrase.endswith('s') and not phrase.endswith('ss') and phrase != "bus"


def add_article(phrase: str) -> str:
    """Automatically add 'a' or 'an' where natural; leave plural or mass nouns unchanged."""
    phrase = phrase.strip()
    first_word = phrase.split()[0].lower()
    vowels = ('a', 'e', 'i', 'o', 'u')
    if is_plural(phrase):
        return phrase  # plural form, no article
    if first_word[0] in vowels:
        return f"an {phrase}"
    return f"a {phrase}"


def is_there(phrase: str) -> str:
    """'Is there a chair' / 'Are there earrings'."""
    return f"{'Are' if is_plural(phrase) else 'Is'} there {add_article(phrase)}"


def default_question(object_name, caption):
    return f"{caption} {is_there(object_name)} in this image? Answer with just yes or no."


def ask_qwen(qwen_model_class, images, questions):
    """Ask Qwen one yes/no question per (PIL image, question) pair, in batches. Returns a list of bools."""
    answers = []
    for start in range(0, len(questions), MAX_BATCH):
        answers += ask_qwen_batch(qwen_model_class, images[start:start + MAX_BATCH], questions[start:start + MAX_BATCH])
    return answers


@torch.no_grad()
def ask_qwen_batch(qwen_model_class, images, questions):
    inputs = qwen_model_class.preprocessor(
        instruction=questions,
        image_file=images,
        response=[""] * len(questions),
        generation_mode=True,
        continue_final_message=False,
    )
    out = qwen_model_class.generate(
        **inputs,
        max_new_tokens=QWEN_MAX_NEW_TOKENS,
        do_sample=False,
    )
    input_len = inputs["input_ids"].shape[1]  # same for every row with left padding
    answers = qwen_model_class.get_tokenizer().batch_decode(out[:, input_len:], skip_special_tokens=True)

    return ["yes" in answer.lower() for answer in answers]


def best_dino_boxes(dino_model, dino_processor, image, object_names, device):
    """For each object, (score, xyxy box) of DINO's most confident box, or None if nothing passes the thresholds."""
    best_boxes = []
    for start in range(0, len(object_names), MAX_BATCH):
        best_boxes += best_dino_boxes_batch(dino_model, dino_processor, image, object_names[start:start + MAX_BATCH], device)
    return best_boxes


@torch.no_grad()
def best_dino_boxes_batch(dino_model, dino_processor, image, object_names, device):
    # Grounding DINO expects lowercase text ending with a period
    texts = [f"{object_name.lower().strip()}." for object_name in object_names]
    inputs = dino_processor(images=[image] * len(texts), text=texts, padding=True, return_tensors="pt").to(device)
    outputs = dino_model(**inputs)
    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=DINO_BOX_THRESH,
        text_threshold=DINO_TEXT_THRESH,
        target_sizes=[(image.height, image.width)] * len(texts),
    )

    best_boxes = []
    for result in results:
        # Keep only boxes whose matched text is not empty
        keep = [i for i, label in enumerate(result["text_labels"]) if label.strip()]
        if not keep:
            best_boxes.append(None)
            continue
        best = max(keep, key=lambda i: result["scores"][i].item())
        best_boxes.append((result["scores"][best].item(), result["boxes"][best].tolist()))
    return best_boxes


def crop(image, box):
    x1, y1, x2, y2 = (int(v) for v in box)
    return image.crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)))  # avoid an empty crop on degenerate boxes


def check_absent(dino_model, dino_processor, qwen_model_class, image, object_names, caption, device):
    """
    Returns one check record per object. Its "result" is "absent" only if the object is confirmed absent;
    otherwise it says which step found it may be there: "dino_confident", "qwen_full_yes" or "qwen_crop_yes".
    """
    best_boxes = best_dino_boxes(dino_model, dino_processor, image, object_names, device)
    checks = [
        {"object": object_name, "dino_score": None if b is None else round(b[0], 3), "result": "absent"}
        for object_name, b in zip(object_names, best_boxes)
    ]

    # DINO confident -> may be there. Otherwise ask Qwen on the full image, even when DINO found nothing,
    # since DINO misses some real objects
    to_ask = []
    for i, b in enumerate(best_boxes):
        if b is not None and b[0] > DINO_CONFIDENT_THRESH:
            checks[i]["result"] = "dino_confident"
        else:
            to_ask.append(i)
    answers = ask_qwen(qwen_model_class, [image] * len(to_ask), [default_question(object_names[i], caption) for i in to_ask])
    for i, yes in zip(to_ask, answers):
        if yes:
            checks[i]["result"] = "qwen_full_yes"

    # If DINO found a box, Qwen must also say no on it, which catches small objects
    with_box = [i for i in to_ask if checks[i]["result"] == "absent" and best_boxes[i] is not None]
    crops = [crop(image, best_boxes[i][1]) for i in with_box]
    answers = ask_qwen(qwen_model_class, crops, [default_question(object_names[i], caption) for i in with_box])
    for i, crop_image, yes in zip(with_box, crops, answers):
        checks[i]["crop_size"] = list(crop_image.size)  # logged to see whether small crops make Qwen say yes
        if yes:
            checks[i]["result"] = "qwen_crop_yes"

    return checks


def first_absent(dino_model, dino_processor, qwen_model_class, image, candidates, caption, device):
    """
    The first candidate (in order) confirmed absent, or None, and the check records of the candidates
    tried before it (the skipped ones) and of itself. Checks MAX_BATCH candidates at a time.
    """
    log = []
    for start in range(0, len(candidates), MAX_BATCH):
        chunk = candidates[start:start + MAX_BATCH]
        for check in check_absent(dino_model, dino_processor, qwen_model_class, image, chunk, caption, device):
            log.append(check)
            if check["result"] == "absent":
                return check["object"], log
    return None, log


def compute_pmi(images_objects, smoothing):
    """
    PMI between every pair of objects, from their co-occurrence over images. Pairs that never co-occur get 0.
    `smoothing` is added to each object's count: plain PMI gives an object seen in a single image the highest
    possible score with everything it appeared with once, so rare (often odd or broken) labels would win;
    smoothing barely changes common objects' scores but lowers those of rare ones.
    """
    total_images = len(images_objects)
    obj_freq = defaultdict(int)
    cooccur = defaultdict(int)
    for objects in images_objects:
        unique_objs = set(objects)
        for obj in unique_objs:
            obj_freq[obj] += 1
        for a, b in combinations(unique_objs, 2):
            cooccur[frozenset([a, b])] += 1

    def pmi(a, b):
        count = cooccur.get(frozenset([a, b]), 0)
        if count == 0:
            return 0
        return math.log(count * total_images / ((obj_freq[a] + smoothing) * (obj_freq[b] + smoothing)))

    vocab = sorted(obj_freq)
    return vocab, {a: {b: pmi(a, b) for b in vocab} for a in vocab}


def related_objects(scene_objects, candidates, pmi_matrix, top_k):
    """Candidates ranked by their total positive PMI with the scene objects."""
    scores = {c: sum(max(pmi_matrix[c][obj], 0) for obj in scene_objects) for c in candidates}
    return sorted(scores, key=lambda c: scores[c], reverse=True)[:top_k]


class ImageDataset(Dataset):
    def __init__(self, images, image_dir):
        self.images = images
        self.image_dir = image_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        return img_info["id"], Image.open(os.path.join(self.image_dir, img_info["file_name"])).convert("RGB")


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

    num_images, samples = 0, []
    absent_results, random_results, crop_yes_sizes, crop_no_sizes = Counter(), Counter(), [], []
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                row = json.loads(line)
                num_images += 1
                if row["sample"] is not None:
                    samples.append(row["sample"])
                absent_results.update(check["result"] for check in row["absent_checks"])
                random_results.update(check["result"] for check in row["random_checks"])
                for check in row["absent_checks"] + row["random_checks"]:
                    if "crop_size" in check:
                        size = check["crop_size"][0] * check["crop_size"][1]
                        (crop_yes_sizes if check["result"] == "qwen_crop_yes" else crop_no_sizes).append(size)

    output_path = os.path.join(save_dir, save_filename + ".json")
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Merged {len(paths)} parts -> {output_path} ({len(samples)} samples from {num_images} images)")

    # Why candidates were skipped: which check said the object may be there
    print(f"Absent-object candidates checked: {dict(absent_results)}")
    print(f"Random-object candidates checked: {dict(random_results)}")
    median = lambda sizes: sorted(sizes)[len(sizes) // 2] if sizes else None
    print(f"Crops asked to Qwen: {len(crop_yes_sizes)} yes (median {median(crop_yes_sizes)} px), "
          f"{len(crop_no_sizes)} no (median {median(crop_no_sizes)} px)")


def load_excluded_objects(paths):
    """img_id (str) -> every object listed for that image in the given files, under any key (kept or rejected)."""
    excluded = defaultdict(set)
    for path in paths:
        with open(PROJECT_ROOT / path, "r", encoding="utf-8") as f:
            for per_image in json.load(f).values():
                for img_id, objects in per_image.items():
                    excluded[img_id].update(objects)
    return excluded


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_files_only", type=bool, default=True)
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR, help="Hugging Face cache folder holding the Qwen and Grounding DINO models.")
    parser.add_argument("--coco_dir", type=str, default=COCO_DIR, help="COCO root folder (must contain annotations/captions_train2017.json and train2017/).")

    # Relative paths below are resolved against the project root
    parser.add_argument("--caption_file", type=str, default="data_files/qwen_captions.json", help="Output of 1_qwen_captioning.py.")
    parser.add_argument("--objects_file", type=str, default="data_files/florence_dino_filtered_objects.json", help="Output of 3_1 or 3_2.")
    parser.add_argument("--objects_name", type=str, default="coco_florence_objects", help="coco_florence_objects (3_1) or coco_instances_objects (3_2).")
    parser.add_argument("--exclude_objects_files", type=str, nargs="+",
                        default=["data_files/florence_dino_filtered_objects.json", "data_files/instances_dino_filtered_objects.json"],
                        help="Objects listed for an image in any of these files (kept or rejected) are never its absent object.")
    parser.add_argument("--save_dir", type=str, default="data_files")
    parser.add_argument("--save_filename", type=str, default=None, help="Default: {objects_name without _objects}_adversary_random_present_objects")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random objects.")
    parser.add_argument("--min_objects", type=int, default=3, help="Skip images with this many present objects or fewer (same as 2_filtering_samples.py).")
    parser.add_argument("--pmi_smoothing", type=float, default=5, help="Added to each object's count in PMI, so rarely seen objects need more evidence to rank high.")

    parser.add_argument("--num_workers", type=int, default=4)

    # Multi-node: launch num_shards jobs (e.g. a SLURM array), each with its own shard_id.
    # Each job is further split across its own GPUs by accelerate.
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)

    # Merge mode: combine the part files of a finished multi-GPU run into the final file
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--num_parts", type=int, default=None, help="With --merge: total number of GPUs the run used.")
    args = parser.parse_args()

    save_dir = PROJECT_ROOT / args.save_dir  # an absolute path overrides PROJECT_ROOT
    save_filename = args.save_filename or args.objects_name.removesuffix("_objects") + "_adversary_random_present_objects"

    if args.merge:
        merge_parts(save_dir, save_filename, args.num_parts)
        raise SystemExit

    annotation_file = os.path.join(args.coco_dir, "annotations", "captions_train2017.json")
    image_dir = os.path.join(args.coco_dir, "train2017")

    accelerator = Accelerator()
    device = accelerator.device

    # Each GPU across all jobs gets a disjoint slice of the images
    rank = args.shard_id * accelerator.num_processes + accelerator.process_index
    world_size = args.num_shards * accelerator.num_processes

    os.makedirs(save_dir, exist_ok=True)
    save_path = part_path(save_dir, save_filename, rank, world_size)

    # Resume: skip images already processed in this GPU's part file
    done_ids = set()
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            done_ids = {json.loads(line)["img_id"] for line in f if line.strip()}

    with open(PROJECT_ROOT / args.objects_file, "r", encoding="utf-8") as f:
        img_id_to_objects = json.load(f)[args.objects_name]  # keys are str(img_id)
    excluded_objects = load_excluded_objects(args.exclude_objects_files)

    id_to_caption = {}
    with open(PROJECT_ROOT / args.caption_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                id_to_caption[item["id"]] = item["prediction"].strip()

    with open(annotation_file, "r") as f:
        coco_data = json.load(f)
    img_id_to_coco_captions = defaultdict(list)
    for ann in coco_data["annotations"]:
        img_id_to_coco_captions[ann["image_id"]].append(ann["caption"])

    # PMI over all images, before sharding, so every GPU uses the same statistics
    vocab, pmi_matrix = compute_pmi(list(img_id_to_objects.values()), args.pmi_smoothing)

    images = [
        img for img in coco_data["images"]
        if len(img_id_to_objects.get(str(img["id"]), [])) > args.min_objects and img["id"] in id_to_caption
    ]
    images = [img for img in images[rank::world_size] if str(img["id"]) not in done_ids]
    img_id_to_filename = {img["id"]: img["file_name"] for img in images}
    print(f"[rank {rank}/{world_size}] {len(images)} images to process ({len(done_ids)} already done) -> {save_path}")

    # batch_size=None: one image at a time; its candidates are batched inside check_absent
    loader = DataLoader(ImageDataset(images, image_dir), batch_size=None, num_workers=args.num_workers)

    qwen_model_class = get_model_class(
        model_name_or_path=QWEN_MODEL,
        model_base="",
        processor_name=None,
        device=device,
        logger=None,
        args=args,
    )
    # Batched generation needs left padding so every prompt ends right before the generated tokens
    qwen_model_class.get_tokenizer().padding_side = "left"

    dino_processor = AutoProcessor.from_pretrained(DINO_MODEL, cache_dir=args.cache_dir)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL, cache_dir=args.cache_dir).to(device).eval()

    with open(save_path, "a") as f:
        for img_id, image in tqdm(loader, desc=f"rank {rank}"):
            key = str(img_id)
            scene_objects = img_id_to_objects[key]
            caption = id_to_caption[img_id]
            candidates = [o for o in vocab if o not in excluded_objects[key] and o not in scene_objects]

            absent_object, absent_checks = first_absent(
                dino_model, dino_processor, qwen_model_class, image,
                related_objects(scene_objects, candidates, pmi_matrix, TOP_K), caption, device,
            )

            random_object, random_checks = None, []
            if absent_object is not None:
                # Seeded per image, so the draw does not depend on how images are split across GPUs
                rng = random.Random(f"{args.seed}-{img_id}")
                random_candidates = rng.sample([o for o in candidates if o != absent_object], NUM_RANDOM_CANDIDATES)
                random_object, random_checks = first_absent(dino_model, dino_processor, qwen_model_class, image, random_candidates, caption, device)

            sample = None
            if random_object is not None:
                sample = {
                    "img_id": key,
                    "img_filename": img_id_to_filename[img_id],
                    "captions": img_id_to_coco_captions[img_id],
                    # the present object most related to the absent one
                    "present_object": max(scene_objects, key=lambda o: pmi_matrix[o][absent_object]),
                    "absent_object": absent_object,
                    "random_object": random_object,
                    "all_present_objects": scene_objects,
                }
            # The checks are kept in the part files only, to see why candidates were skipped (summarized by --merge)
            f.write(json.dumps({"img_id": key, "sample": sample, "absent_checks": absent_checks, "random_checks": random_checks}) + "\n")
            f.flush()  # so a killed job keeps what it finished

    open(save_path + ".done", "w").close()

    if world_size == 1:
        merge_parts(save_dir, save_filename, num_parts=1)
