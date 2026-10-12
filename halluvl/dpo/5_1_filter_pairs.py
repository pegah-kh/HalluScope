import argparse
import os

import torch.nn.functional as F

from utils import assistant_answer, embed, load_model_outputs, write_json

CPQ_REJECTED_REFUSALS = ["cannot", "no visible", "not visible", "does not show", "there is no",
                         "not present in the image", "there are no"]

TPQ_REJECTED_NEGATIONS = {
    "qwen": ["cannot", "no visible", "not visible", "does not show", "does not contain", "does not depict",
             "does not apply", "there is no", "there are no", "not present in the image", " not ", " no ",
             "not relevant"],
    "llava": ["cannot", "no visible", "not visible", "does not show", "there is no", "not present in the image",
              " not ", " no ", "not relevant"],
}


def keep_pair(query_type, chosen, rejected, model):
    if query_type == "cpq":
        return (not any(p in rejected.lower() for p in CPQ_REJECTED_REFUSALS)
                and (" no " in chosen or " not " in chosen))
    if query_type == "tpq":
        return any(p in rejected.lower() for p in TPQ_REJECTED_NEGATIONS[model])
    if query_type == "pope":
        return not ("there is no" in chosen.lower() or "no, " in chosen.lower() or "yes" in rejected.lower())
    if query_type == "pope_negative":
        return not ("there is no" in rejected.lower() or "no, " in rejected.lower() or "yes" in chosen.lower())
    return True


def user_question(instruction, query_type):
    """Recover the question without its hint from the instruction of the hinted (or plain) prompt."""
    instruction = instruction.split("assistant")[0]
    if query_type == "description":
        question = instruction.split(". ")[-1].split(", ")[-1]
    else:
        parts = instruction.split(". ")
        question = parts[1] if len(parts) > 1 else parts[0]
    return question[0].upper() + question[1:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pair grounded (chosen) and hallucinated (rejected) answers and keep dissimilar pairs.")
    parser.add_argument("--type", choices=["cpq", "tpq", "pope", "pope_negative", "description"], required=True,
                        help="'description' covers the unilateral, contrastive and hallucination-injection strategies.")
    parser.add_argument("--model", choices=["qwen", "llava"], required=True)
    parser.add_argument("--chosen", required=True, help="Answers (4_1_generate_answers.py) giving the chosen side.")
    parser.add_argument("--rejected", required=True, help="Answers (4_1_generate_answers.py) giving the rejected side.")
    parser.add_argument("--threshold", type=float, default=0.9, help="Drop pairs whose BGE-M3 cosine similarity is above this.")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    chosen_preds, chosen_instructions, image_paths, image_ids = load_model_outputs(args.chosen)
    rejected_preds, rejected_instructions, _, rejected_ids = load_model_outputs(args.rejected)
    instructions = chosen_instructions if args.type == "tpq" else rejected_instructions
    if args.type == "description":
        assert image_ids == rejected_ids
    chosen_preds = [assistant_answer(p, args.model) for p in chosen_preds]
    rejected_preds = [assistant_answer(p, args.model) for p in rejected_preds]

    similarities = F.cosine_similarity(embed(rejected_preds), embed(chosen_preds), dim=1).tolist()

    pairs = []
    for image_id, image_path, instruction, chosen, rejected, similarity in zip(
            image_ids, image_paths, instructions, chosen_preds, rejected_preds, similarities):
        if similarity > args.threshold or not keep_pair(args.type, chosen, rejected, args.model):
            continue
        pairs.append({"id": int(image_id), "image": os.path.basename(image_path),
                      "prompt": f"<image>\n{user_question(instruction, args.type)}", "chosen": chosen, "rejected": rejected})

    print(f"Kept {len(pairs)} / {len(image_ids)} pairs")
    write_json(pairs, args.output_path)
