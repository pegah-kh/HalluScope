import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel
from tqdm import tqdm

from utils import assistant_answer, gpt_parse, load_model_outputs, write_jsonl


class IncorrectAnswer(BaseModel):
    incorrect_answer: str


INSTRUCTION = """
    Given a question about an image and its correct answer, generate a **plausible but incorrect answer**.  

    Rules:
    1. The wrong answer should relate to the question and contradict the correct answer.  
    2. Keep it realistic and similar in style/detail to the correct answer.  
    3. Avoid nonsense or unrelated text.  

    Example:
    question: "what colors are the lights illuminating the skyscrapers?"  
    correct answer: "The lights illuminating the skyscrapers in the image are primarily blue and white."  
    wrong answer: "The lights illuminating the skyscrapers in the image are primarily red and yellow."

    Now generate a wrong answer for:

    question: "{question}"  
    correct answer: "{answer}"
    """


def invert_answer(sample):
    parsed = gpt_parse(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": INSTRUCTION.format(question=sample["question"], answer=sample["answer"])},
        ],
        response_format=IncorrectAnswer,
    )
    return {"attribute_question": sample["question"], "right_answer": sample["answer"],
            "wrong_answer": parsed.incorrect_answer, "id": sample["id"], "image": sample["image"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model-assisted answer inversion: GPT writes a wrong-attribute answer to each TPQ answer.")
    parser.add_argument("--tpq_outputs", required=True, help="Answers (4_1_generate_answers.py) to the plain TPQ prompts.")
    parser.add_argument("--model", choices=["qwen", "llava"], required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_workers", type=int, default=20)
    args = parser.parse_args()

    predictions, instructions, image_paths, image_ids = load_model_outputs(args.tpq_outputs)
    samples = [{"id": image_id, "image": os.path.basename(path), "question": instruction,
                "answer": assistant_answer(pred, args.model)}
               for pred, instruction, path, image_id in zip(predictions, instructions, image_paths, image_ids)]

    answers = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(invert_answer, sample) for sample in samples]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                answers.append(future.result())
            except Exception as e:
                print(f"Sample failed: {e}")

    write_jsonl(answers, args.output_path)
