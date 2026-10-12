import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel
from tqdm import tqdm

from utils import gpt_parse, read_json, write_jsonl


class CaptionTriplet(BaseModel):
    caption: str
    prompt: str
    prompt_neg: str
    present: list[str]
    no_exist: list[str]


class CaptionTriplets(BaseModel):
    list: list[CaptionTriplet]


SYSTEM_INSTRUCTION = """Given multiple captions, return a single JSON array containing one object per caption.
The array format MUST be exactly:
[
  {"caption": "...", "present": [...], "no-exist": [...], "prompt": "...", "prompt_neg": "..."},
  {"caption": "...", "present": [...], "no-exist": [...], "prompt": "...", "prompt_neg": "..."}
]

Instructions:
1. Rewrite each caption as a clear, concise Stable Diffusion prompt that describes only specific, concrete visual objects and scenes. Avoid abstract concepts, emotions, or vague descriptions, focusing solely on tangible, observable elements..
2. 'present': list CONCRETE objects explicitly mentioned in the caption.
3. 'no-exist': list CONCRETE objects not in the caption but likely in similar scenes.
4. 'prompt': embed prompt weights for all 'present' objects (1.0–1.5).
5. 'prompt_neg': embed weights for all 'no-exist' objects.
6. Always include at least one object in each list.


Example: 
original caption: 'Two wild pheasants standing at the edge of a dense forest, with autumn foliage and dappled sunlight.'
output json:
[{
  "caption": "Two wild pheasants standing at the edge of a dense forest with autumn leaves and sunlight filtering through the trees.",
  "present": ["pheasants", "forest", "leaves", "trees"],
  "no-exist": ["road", "house", "fence"],
  "prompt": "Two wild (pheasants:1.5) standing at the edge of a dense (forest:1.3) with autumn (leaves:1.2) and sunlight filtering through the (trees:1.2).",
  "prompt_neg": "(road:1.3), (house:1.3), (fence:1.2)"
}]


Return only the JSON output, no explanations or text.
"""


def generate_triplets(captions):
    caption_block = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(captions))
    parsed = gpt_parse(
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"Captions:\n{caption_block}"},
        ],
        response_format=CaptionTriplets,
    )
    return [item.model_dump() for item in parsed.list]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewrite captions into SD prompts with present / absent object lists.")
    parser.add_argument("--caption_pool", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--total_num", type=int, default=15000)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_workers", type=int, default=20)
    args = parser.parse_args()

    captions = read_json(args.caption_pool)
    n_batches = min(args.total_num, len(captions)) // args.batch_size

    triplets = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(generate_triplets, captions[i * args.batch_size:(i + 1) * args.batch_size])
                   for i in range(n_batches)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                triplets.extend(future.result())
            except Exception as e:
                print(f"Batch failed: {e}")

    # The line number of each triplet is its image index in all later steps.
    write_jsonl(triplets, args.output_path)
