"""
This code is to filter the Florence object detection results using Grounding DINO and Qwen.
The goal is to check that the detected objects are actually present in the image, and filter out false positives.

For each (image, object) pair:
1. Grounding DINO (tiny) looks for the object. No box above the thresholds -> Qwen is asked a strict
   question on the full image (yes only if the object is clearly visible and really is that object),
   without the caption. A "yes" keeps the object.
2. If DINO's best box is very confident (score > DINO_CONFIDENT_THRESH) -> object kept.
3. Otherwise Qwen is asked "{caption} Is there a/an {object} in this image?" on the full image,
   then on DINO's best box crop. A "yes" on either keeps the object.
All objects of an image go through each step together, in one batch.

Output: data_files/florence_dino_filtered_objects.json = {
    "coco_florence_objects": {img_id: [kept objects]},
    "rejected_objects": {img_id: [rejected objects]},
}
Rejected objects are uncertain (Florence saw something the checks could not confirm), so they must be
used neither as present objects nor as absent (adversarial) objects for that image.

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
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

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
DINO_CONFIDENT_THRESH = 0.6  # above this, trust DINO without asking Qwen
QWEN_MAX_NEW_TOKENS = 16  # only a yes/no answer is needed
MAX_BATCH = 8  # max objects per DINO / Qwen batch, bounds GPU memory on images with many objects

# Florence labels are cleaned before checking, and kept cleaned in the output.
LABEL_RENAMES = {
    # Open Images label names that DINO and Qwen handle badly
    "kitchen & dining room table": "table",
    "coffee cup": "cup",
    "vehicle registration plate": "license plate",
    "studio couch": "couch",
    "trash bin/can": "trash can",
    "power plugs and sockets": "power outlet",
    "fancy dress": "dress",  # Florence uses it for any dress, not for costumes
    "bow (Decorative ribbon)": "ribbon bow",
    "bow (decorative ribbons)": "ribbon bow",
    # Florence generation glitches with an obvious intended word
    "loudoudspeakers": "loudspeaker",
    "vanvan": "van",
    "bottlele cap": "bottle cap",
    "plasticastic bag": "plastic bag",
    "tentennis racket": "tennis racket",
    "trafficffic light": "traffic light",
    "lampsshade": "lampshade",
    "desK": "desk",
    "teddybear": "teddy bear",
    "towels/napskin": "towel/napkin",
    "bull mastif": "bull mastiff",
    "salt-and- pepper shippers": "salt and pepper shakers",
    # Spelling or singular/plural variants, merged into the more frequent form
    "salt & pepper shaker": "salt and pepper shakers",
    "computer mouse pad": "mousepad",
    "bottle caps": "bottle cap",
    "bottles cap": "bottle cap",
    "plastic bags": "plastic bag",
    "candles holder": "candle holder",
    "towels rack": "towel rack",
    "towels/napkin": "towel/napkin",
    "towels": "towel",
    "tennis balls": "tennis ball",
    "picture frames": "picture frame",
    "pigeons": "pigeon",
    "refrigerators": "refrigerator",
    "miniskirts": "miniskirt",
    "skateboards": "skateboard",
    "doughnuts": "doughnut",
    "trouser": "trousers",
    "sneaker": "sneakers",
    "boots": "boot",
}
# Florence generation glitches without a clear meaning: never checked, never kept
LABEL_DROPS = {
    "lampplant", "sauver", "frisver", "loudscrewdriver", "lampscrewdriver",
    "loudshade", "lampstick", "salt-and-pear", "banana cap", "lamp cap",
}


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


def strict_question(object_name):
    # Used when DINO found nothing. No caption: it was written by Qwen and could push it towards yes
    it_is, it = ("they are", "them") if is_plural(object_name) else ("it is", "it")
    return (
        f"{is_there(object_name)} clearly visible in this image? "
        f"Answer yes only if {it_is} clearly visible with most of {it} in view "
        f"(not mostly cut off by the image border or hidden behind other things), "
        f"and {it_is.replace(' ', ' really ')} {add_article(object_name)}, not a similar object or a picture of one. "
        f"Answer with just yes or no."
    )


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


def check_objects(dino_model, dino_processor, qwen_model_class, image, object_names, caption, device):
    """Returns one bool per object: True if the object is kept."""
    best_boxes = best_dino_boxes(dino_model, dino_processor, image, object_names, device)
    kept = [False] * len(object_names)

    # 1. DINO found nothing: DINO misses some clearly visible objects, so give Qwen a second, strict look
    no_box = [i for i, b in enumerate(best_boxes) if b is None]
    answers = ask_qwen(qwen_model_class, [image] * len(no_box), [strict_question(object_names[i]) for i in no_box])
    for i, yes in zip(no_box, answers):
        kept[i] = yes

    # 2. DINO is confident
    for i, b in enumerate(best_boxes):
        if b is not None and b[0] > DINO_CONFIDENT_THRESH:
            kept[i] = True

    # 3. DINO is unsure: ask Qwen on the full image, then on DINO's best box,
    # which helps when the object is present but small in the image
    unsure = [i for i, b in enumerate(best_boxes) if b is not None and b[0] <= DINO_CONFIDENT_THRESH]
    answers = ask_qwen(qwen_model_class, [image] * len(unsure), [default_question(object_names[i], caption) for i in unsure])
    for i, yes in zip(unsure, answers):
        kept[i] = yes

    still_unsure = [i for i, yes in zip(unsure, answers) if not yes]
    answers = ask_qwen(
        qwen_model_class,
        [crop(image, best_boxes[i][1]) for i in still_unsure],
        [default_question(object_names[i], caption) for i in still_unsure],
    )
    for i, yes in zip(still_unsure, answers):
        kept[i] = yes

    return kept


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

    image_to_objects, image_to_rejected = {}, {}
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                row = json.loads(line)
                if row["kept"]:
                    image_to_objects[row["id"]] = row["kept"]
                if row["rejected"]:
                    image_to_rejected[row["id"]] = row["rejected"]

    output_path = os.path.join(save_dir, save_filename + ".json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "coco_florence_objects": image_to_objects,
            "rejected_objects": image_to_rejected,
        }, f, indent=2, ensure_ascii=False)
    print(f"Merged {len(paths)} parts -> {output_path} ({len(image_to_objects)} images with kept objects)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_files_only", type=bool, default=True)
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR, help="Hugging Face cache folder holding the Qwen and Grounding DINO models.")
    parser.add_argument("--coco_dir", type=str, default=COCO_DIR, help="COCO root folder (must contain annotations/captions_train2017.json and train2017/).")

    # Relative paths below are resolved against the project root
    parser.add_argument("--filtered_metadata_path", type=str, default="data_files/selected_coco_metadata.json", help="Output of step 1_diverse_caption_pool.")
    parser.add_argument("--caption_file", type=str, default="data_files/qwen_captions.json", help="Output of 1_qwen_captioning.py.")
    parser.add_argument("--florence_id_to_objects_path", type=str, default="data_files/coco_florence_objects.json", help="Output of 2_florence_objs_detection.py.")
    parser.add_argument("--save_dir", type=str, default="data_files")
    parser.add_argument("--save_filename", type=str, default="florence_dino_filtered_objects")

    parser.add_argument("--num_workers", type=int, default=4)

    # Multi-node: launch num_shards jobs (e.g. a SLURM array), each with its own shard_id.
    # Each job is further split across its own GPUs by accelerate.
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)

    # Merge mode: combine the part files of a finished multi-GPU run into the final JSON
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--num_parts", type=int, default=None, help="With --merge: total number of GPUs the run used.")
    args = parser.parse_args()

    save_dir = PROJECT_ROOT / args.save_dir  # an absolute path overrides PROJECT_ROOT

    if args.merge:
        merge_parts(save_dir, args.save_filename, args.num_parts)
        raise SystemExit

    annotation_file = os.path.join(args.coco_dir, "annotations", "captions_train2017.json")
    image_dir = os.path.join(args.coco_dir, "train2017")

    accelerator = Accelerator()
    device = accelerator.device

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

    with open(PROJECT_ROOT / args.filtered_metadata_path, "r", encoding="utf-8") as f:
        filtered_img_ids = set(json.load(f)["img_ids"])

    id_to_caption = {}
    with open(PROJECT_ROOT / args.caption_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                id_to_caption[item["id"]] = item["prediction"].strip()

    with open(PROJECT_ROOT / args.florence_id_to_objects_path, "r", encoding="utf-8") as f:
        id_to_objects = json.load(f)["coco_florence_objects"]  # keys are str(img_id)

    with open(annotation_file, "r") as f:
        images = [img for img in json.load(f)["images"] if img["id"] in filtered_img_ids and str(img["id"]) in id_to_objects]

    images = [img for img in images[rank::world_size] if img["id"] not in done_ids]
    print(f"[rank {rank}/{world_size}] {len(images)} images to process ({len(done_ids)} already done) -> {save_path}")

    # batch_size=None: one image at a time; its objects are batched inside check_objects
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
            # dict.fromkeys drops duplicates a rename could create, keeping the order
            objects = list(dict.fromkeys(LABEL_RENAMES.get(o, o) for o in id_to_objects[str(img_id)] if o not in LABEL_DROPS))
            if not objects:
                continue
            kept = check_objects(dino_model, dino_processor, qwen_model_class, image, objects, id_to_caption[img_id], device)

            f.write(json.dumps({
                "id": img_id,
                "kept": [o for o, k in zip(objects, kept) if k],
                "rejected": [o for o, k in zip(objects, kept) if not k],
            }) + "\n")
            f.flush()  # so a killed job keeps what it finished

    open(save_path + ".done", "w").close()

    if world_size == 1:
        merge_parts(save_dir, args.save_filename, num_parts=1)
