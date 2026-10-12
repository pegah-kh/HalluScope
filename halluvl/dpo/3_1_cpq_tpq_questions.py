import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from tqdm import tqdm

from utils import embed, gpt_parse, read_jsonl, write_jsonl


class CPQ(BaseModel):
    hall_question: str
    hall_object: str
    truth_question: str
    truth_object: str


class CPQList(BaseModel):
    list_cpq: list[CPQ]


INSTRUCTION = """
    Given the JSON array input below, generate a JSON array output in this format:
    [
    {{"hall_question": "xxx", "hall_object": "xxx", "truth_question": "xxx", "truth_object": "xxx"}},
    {{"hall_question": "xxx", "hall_object": "xxx", "truth_question": "xxx", "truth_object": "xxx"}},
    ...
    ]

    Each JSON in the output should satisfy the following rules:
    1. **hall_question:** Ask about an object from the 'no_exist' list that seems most likely to appear in the caption. The question must assume the object is present and should avoid trivial, generic, or yes/no phrasing.
    2. **hall_object:** The exact object from 'no_exist' used in hall_question.
    3. **truth_question:** Ask a natural, visually grounded question about one object from the 'present' list. Assume only its visibility, and avoid questions that rely on other objects, context, or common knowledge.
    4. **truth_object:** The exact object from 'present' used in truth_question.
    5. Avoid starting questions with "Where", "How many", or "Is there" unless absolutely necessary.
    6. hall_object and truth_object must appear verbatim in their respective questions.
    7. Avoid generating questions similar to those listed in 'avoid_questions'.
    8. The output must be valid JSON only — no extra text, formatting, or commentary.

    Example:
    Input:
    [{{"caption": "a young woman is walking along the beach during sunset", "present": ["woman", "beach", "sunset"], "no_exist": ["umbrella", "dog", "seagulls"]}}]

    Output:
    [{{"hall_question": "What are the seagulls doing in the image?", "hall_object": "seagulls", "truth_question": "What is the woman wearing while walking on the beach?", "truth_object": "woman"}}]

    Now generate the JSON array output for:
    {items}
    """


class QuestionMemory:
    """Hallucination questions already generated, indexed by caption embedding, to discourage repetitive questions."""

    def __init__(self, threshold=0.8, min_size=500, top_k=3):
        self.questions = {}
        self.embeddings = None
        self.threshold = threshold
        self.min_size = min_size
        self.top_k = top_k

    def similar(self, embedding):
        if self.embeddings is None or len(self.questions) < self.min_size:
            return []
        similarities = F.cosine_similarity(embedding, self.embeddings)
        above = (similarities > self.threshold).nonzero(as_tuple=True)[0]
        order = similarities[above].argsort(descending=True)
        return [self.questions[tuple(self.embeddings[above[i]].numpy())] for i in order[:self.top_k]]

    def add(self, embedding, question):
        key = tuple(embedding.numpy())
        if key in self.questions:
            return
        self.questions[key] = question
        row = embedding.unsqueeze(0)
        self.embeddings = row if self.embeddings is None else torch.cat((self.embeddings, row))


def generate_questions(items, memory):
    embeddings = []
    for item in items:
        embedding = embed([item["caption"]])[0]
        embeddings.append(embedding)
        similar = memory.similar(embedding)
        item["avoid_questions"] = "\n".join(json.dumps(q) for q in similar) if similar else "None"

    parsed = gpt_parse(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": INSTRUCTION.format(items=str(items))},
        ],
        response_format=CPQList,
    )

    outputs = []
    for question, item, embedding in zip(parsed.list_cpq, items, embeddings):
        outputs.append({**question.model_dump(), "idx": item["idx"]})
        memory.add(embedding, question.hall_question)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate one counterfactual (CPQ) and one true (TPQ) presupposition question per image.")
    parser.add_argument("--verified", required=True, help="Output of 2_3_filter_detections.py.")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_workers", type=int, default=20)
    args = parser.parse_args()

    items = [{"caption": a["caption"], "present": a["present"], "no_exist": a["no_exist"], "idx": a["idx"]}
             for a in read_jsonl(args.verified)]
    memory = QuestionMemory()

    # As in the released run, a trailing incomplete batch is not submitted.
    n_batches = len(items) // args.batch_size
    questions = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(generate_questions, items[i * args.batch_size:(i + 1) * args.batch_size], memory)
                   for i in range(n_batches)]
        for future in tqdm(as_completed(futures), total=n_batches):
            try:
                questions.extend(future.result())
            except Exception as e:
                print(f"Batch failed: {e}")

    write_jsonl(questions, args.output_path)
