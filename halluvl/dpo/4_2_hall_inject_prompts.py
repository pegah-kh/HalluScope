import argparse
import random

from utils import add_article, load_model_outputs, read_jsonl, write_jsonl


def hallucination_prefix(obj):
    obj = add_article(obj)
    return random.choice([
        f"The image shows {obj}",
        f"The scene includes {obj}",
        f"There is {obj} visible in the image",
        f"The picture features {obj}",
        f"In the image, {obj} can be seen",
        f"The photograph depicts {obj}",
        f"The view contains {obj}",
        f"The setting reveals {obj}",
        f"Visible in the image is {obj}",
        f"The composition highlights {obj}",
    ])


def inject_hallucination(description, obj):
    """Replace one sentence (later ones more likely) with a prefix asserting `obj`, and drop everything after it."""
    sentences = [s.strip() for s in description.split(".") if s.strip()]
    n = len(sentences)
    if n == 0:
        raise ValueError("Empty description")
    if n == 1:
        # The released data appends the prefix to the single sentence without a separator.
        return sentences[0] + hallucination_prefix(obj)
    weights = [1 + k / (n - 1) for k in range(n)]
    replaced = random.choices(range(n), weights=[w / sum(weights) for w in weights], k=1)[0]
    sentences[replaced] = hallucination_prefix(obj)
    return ". ".join(sentences[:replaced + 1])


def completion_prompt(prediction, obj, model):
    """Text to be continued by the model in completion mode (no chat template is re-applied)."""
    if model == "llava":
        return inject_hallucination(prediction, obj)
    before, answer = prediction.split("assistant\n")[0].strip(), prediction.split("assistant\n")[1].strip()
    user_turn = before.split("\nuser\n")[1].strip()
    return user_turn + "assistant\n" + inject_hallucination(answer, obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build post-hoc hallucination-injection prompts from the model's own descriptions.")
    parser.add_argument("--verified", required=True, help="Output of 2_3_filter_detections.py.")
    parser.add_argument("--descriptions", required=True, help="Answers (4_1_generate_answers.py) to the plain description prompts.")
    parser.add_argument("--model", choices=["qwen", "llava"], required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    random.seed(0)
    predictions, instructions, _, image_ids = load_model_outputs(args.descriptions)
    outputs = {image_id: (pred, instr) for pred, instr, image_id in zip(predictions, instructions, image_ids)}

    prompts = []
    for item in read_jsonl(args.verified):
        if str(item["idx"]) not in outputs:
            continue
        prediction, instruction = outputs[str(item["idx"])]
        hall_object = item["no_exist"][0]
        prompts.append({
            "image": f"{item['idx']}.jpg",
            "text": instruction,
            "hall_object": hall_object,
            "truth_object": item["present"][0],
            "text_hint": completion_prompt(prediction, hall_object, args.model),
        })
    write_jsonl(prompts, args.output_path)
