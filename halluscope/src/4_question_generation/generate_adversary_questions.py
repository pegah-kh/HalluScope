"""
Generates, for each stage-3 sample, two questions with an LLM (gpt-5-mini by default):
- adversary_question: about the absent object, presupposing it is in the image ("What color is the bow and arrow?");
- truth_question: a visually grounded question about the present object.
The first COCO caption tells the LLM what the scene is. To keep the questions varied, each request lists the
adversary questions already generated for the most similar captions (BGE-M3 embeddings) as questions to avoid.

Samples are sent in batches of BATCH_SIZE. Question pairs are matched to samples by position, so a batch is
retried when the LLM returns a different number of pairs. Results are saved as they come in a progress file, so a
rerun only generates what is missing (e.g. after failed requests); the final file follows the input order.

The LLM is any model behind the OpenAI chat API: OpenAI itself (key in OPENAI_API_KEY), or a model served with an
OpenAI-compatible server (e.g. vLLM), given with --api_base.

Reads:  data_files/coco_{subset}_adversary_random_present_objects_filtered.json (stage 3)
Writes: data_files/coco_{subset}_difficult_samples_adversary_questions.json (JSON Lines):
        the stage-3 fields plus "adversary_question" and "truth_question"
"""
import os
import json
import time
import random
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import openai
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# halluscope/halluscope/, two levels above this file's folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

BATCH_SIZE = 20  # samples per LLM request
EMBEDDING_MODEL = "BAAI/bge-m3"
SIMILARITY_THRESHOLD = 0.8  # captions at least this similar share their questions as questions to avoid
MAX_AVOID_QUESTIONS = 3
MIN_MEMORY_SIZE = 200  # questions to avoid are only given once this many questions exist


class question_pair(BaseModel):
    hall_question: str
    truth_question: str


class question_pair_list(BaseModel):
    pairs: list[question_pair]


def instruction_for(question_jsons):
    # The prompt used for the paper's benchmark; keep it unchanged for comparable questions
    return f"""
    Given the JSON array input below, generate a JSON array output in this format:
    [
    {{"hall_question": "xxx", "truth_question": "xxx"}},
    {{"hall_question": "xxx", "truth_question": "xxx"}},
    ...
    ]

    Each JSON in the output should satisfy the following rules:
    1. **hall_question:** Ask about the object 'no_exist'. The question must assume the object is present and should avoid trivial, generic, or yes/no phrasing.
    2. **truth_question:** Ask a natural, visually grounded question about the 'present' object. Assume only its visibility, and avoid questions that rely on other objects, context, or common knowledge.
    3. Avoid starting questions with "Where", "How many", or "Is there" unless absolutely necessary.
    4. 'no_exist' and 'present' objects must appear verbatim in their respective questions.
    5. Avoid generating questions similar to those listed in 'avoid_questions'.
    6. The output must be valid JSON only — no extra text, formatting, or commentary.

    Example:
    Input:
    [{{"caption": "a young woman is walking along the beach during sunset", "present": "woman", "no_exist": "seagulls"}}]

    Output:
    [{{"hall_question": "What are the seagulls doing in the image?", "truth_question": "What is the woman wearing while walking on the beach?"}}]

    Now generate the JSON array output for:
    {str(question_jsons)}
    """


class QuestionMemory:
    """Adversary questions generated so far, with the embedding of their caption. Shared by the request threads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.embeddings = []
        self.questions = []
        self.seen = set()

    def add(self, embedding, question):
        with self.lock:
            # One question per distinct caption embedding
            key = tuple(embedding.tolist())
            if key not in self.seen:
                self.seen.add(key)
                self.embeddings.append(embedding)
                self.questions.append(question)

    def similar(self, embedding):
        with self.lock:
            if len(self.questions) < MIN_MEMORY_SIZE:
                return []
            similarities = torch.stack(self.embeddings) @ embedding  # embeddings are normalized
            questions = list(self.questions)
        above = (similarities > SIMILARITY_THRESHOLD).nonzero(as_tuple=True)[0]
        best = above[similarities[above].argsort(descending=True)][:MAX_AVOID_QUESTIONS]
        return [questions[i] for i in best.tolist()]


@torch.no_grad()
def embed_captions(captions, batch_size=64):
    """Normalized BGE-M3 ([CLS]) embeddings, on CPU."""
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(device).eval()
    embeddings = []
    for start in tqdm(range(0, len(captions), batch_size), desc="embedding captions"):
        inputs = tokenizer(captions[start:start + batch_size], padding=True, truncation=True, return_tensors="pt").to(device)
        embeddings.append(torch.nn.functional.normalize(model(**inputs)[0][:, 0], p=2, dim=1).cpu())
    return torch.cat(embeddings)


class WrongLengthError(Exception):
    """The LLM returned a different number of question pairs than samples sent."""


RETRIED_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    WrongLengthError,
)


def generate_batch(client, llm_model, items, embeddings, memory, max_retries=10):
    """Returns one (adversary_question, truth_question) pair per item."""
    question_jsons = []
    for item, embedding in zip(items, embeddings):
        avoid = memory.similar(embedding)
        question_jsons.append({
            "caption": item["captions"][0],
            "present": item["present_object"],
            "no_exist": item["absent_object"],
            "avoid_questions": "\n".join(json.dumps(q) for q in avoid) if avoid else "None",
        })

    # Exponential backoff, from https://platform.openai.com/docs/guides/rate-limits
    delay = 1
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.parse(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": instruction_for(question_jsons)},
                ],
                response_format=question_pair_list,
            )
            message = response.choices[0].message
            if getattr(message, "refusal", None):
                raise RuntimeError(f"The LLM refused: {message.refusal}")
            pairs = [(p.hall_question, p.truth_question) for p in message.parsed.pairs]
            if len(pairs) != len(items):
                raise WrongLengthError(f"{len(pairs)} question pairs for {len(items)} samples")
            return pairs
        except RETRIED_ERRORS:
            if attempt == max_retries:
                raise
            delay *= 2 * (1 + random.random())
            time.sleep(delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="florence", choices=["florence", "instances"])
    parser.add_argument("--llm_model", type=str, default="gpt-5-mini")
    parser.add_argument("--api_base", type=str, default=None, help="OpenAI-compatible server, e.g. http://localhost:8000/v1. Default: OpenAI")
    parser.add_argument("--max_workers", type=int, default=20, help="Parallel LLM requests.")

    # Relative paths below are resolved against the project root
    parser.add_argument("--save_dir", type=str, default="data_files")
    parser.add_argument("--input_filename", type=str, default=None, help="Default: data_files/coco_{subset}_adversary_random_present_objects_filtered.json")
    parser.add_argument("--save_filename", type=str, default=None, help="Default: coco_{subset}_difficult_samples_adversary_questions")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None and args.api_base is None:
        raise SystemExit("Set the OPENAI_API_KEY environment variable, or give --api_base for a local LLM.")
    client = OpenAI(api_key=api_key or "EMPTY", base_url=args.api_base)

    save_dir = PROJECT_ROOT / args.save_dir  # an absolute path overrides PROJECT_ROOT
    input_path = PROJECT_ROOT / (args.input_filename or f"data_files/coco_{args.subset}_adversary_random_present_objects_filtered.json")
    save_filename = args.save_filename or f"coco_{args.subset}_difficult_samples_adversary_questions"

    with open(input_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    embeddings = embed_captions([item["captions"][0] for item in items])

    # Resume: questions already generated, by index in the input file
    progress_path = save_dir / f"{save_filename}_progress.jsonl"
    memory = QuestionMemory()
    questions = {}
    if progress_path.exists():
        with open(progress_path, "r") as f:
            for line in f:
                row = json.loads(line)
                questions[row["index"]] = (row["adversary_question"], row["truth_question"])
                memory.add(embeddings[row["index"]], row["adversary_question"])

    todo = [i for i in range(len(items)) if i not in questions]
    batches = [todo[start:start + BATCH_SIZE] for start in range(0, len(todo), BATCH_SIZE)]
    num_failed = 0
    with open(progress_path, "a") as f, ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(generate_batch, client, args.llm_model, [items[i] for i in batch], [embeddings[i] for i in batch], memory): batch
            for batch in batches
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"generating {args.subset}"):
            batch = futures[future]
            try:
                pairs = future.result()
            except Exception as e:
                num_failed += len(batch)
                print(f"Generation failed for {len(batch)} samples, rerun to retry them: {e}")
                continue
            for i, (adversary_question, truth_question) in zip(batch, pairs):
                questions[i] = (adversary_question, truth_question)
                memory.add(embeddings[i], adversary_question)
                f.write(json.dumps({"index": i, "adversary_question": adversary_question, "truth_question": truth_question}, ensure_ascii=False) + "\n")
            f.flush()

    with open(save_dir / f"{save_filename}.json", "w", encoding="utf-8") as f:
        for i, item in enumerate(items):
            if i in questions:
                adversary_question, truth_question = questions[i]
                f.write(json.dumps({**item, "adversary_question": adversary_question, "truth_question": truth_question}, ensure_ascii=False) + "\n")

    print(f"{len(questions)}/{len(items)} samples have questions -> {save_dir / (save_filename + '.json')}")
    if num_failed:
        print(f"Missing: {num_failed} samples from failed requests. Rerun to generate them.")
