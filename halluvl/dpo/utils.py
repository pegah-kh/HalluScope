import json
import random
import time

GPT_MODEL = "gpt-5-mini"
EMBEDDING_MODEL = "BAAI/bge-m3"

_openai_client = None
_embedder = None


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def add_article(phrase):
    phrase = phrase.strip()
    if phrase.endswith("s") and not phrase.endswith("ss"):
        return phrase
    if phrase[0].lower() in "aeiou":
        return f"an {phrase}"
    return f"a {phrase}"


def retry_with_exponential_backoff(func, initial_delay=1, exponential_base=2, max_retries=10):
    # https://platform.openai.com/docs/guides/rate-limits
    def wrapper(*args, **kwargs):
        import openai

        delay = initial_delay
        for _ in range(max_retries):
            try:
                return func(*args, **kwargs)
            except openai.RateLimitError:
                delay *= exponential_base * (1 + random.random())
                time.sleep(delay)
        return func(*args, **kwargs)

    return wrapper


@retry_with_exponential_backoff
def _parse(**kwargs):
    global _openai_client
    import openai

    if _openai_client is None:
        _openai_client = openai.OpenAI()
    return _openai_client.chat.completions.parse(**kwargs)


def gpt_parse(messages, response_format):
    """Structured-output call to GPT; reads OPENAI_API_KEY from the environment."""
    message = _parse(model=GPT_MODEL, messages=messages, response_format=response_format).choices[0].message
    if message.refusal:
        raise RuntimeError("Model refused to produce output")
    if not isinstance(message.parsed, response_format):
        raise TypeError(f"Unexpected parsed type: {type(message.parsed)}")
    return message.parsed


def embed(sentences, batch_size=32):
    """L2-normalised BGE-M3 [CLS] embeddings, returned on CPU."""
    global _embedder
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _embedder is None:
        _embedder = (AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
                     AutoModel.from_pretrained(EMBEDDING_MODEL).to(device).eval())
    tokenizer, model = _embedder

    embeddings = []
    for i in range(0, len(sentences), batch_size):
        inputs = tokenizer(sentences[i:i + batch_size], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            cls = model(**inputs)[0][:, 0]
        embeddings.append(F.normalize(cls, p=2, dim=1).cpu())
    return torch.cat(embeddings)


def load_model_outputs(path):
    """Read the answers written by 4_1_generate_answers.py: predictions, instructions, image paths and image ids.

    Qwen2-VL predictions are the full decoded chat ("system\\n...user\\n...assistant\\n<answer>"),
    LLaVA-1.5 predictions are the answer only.
    """
    rows = read_jsonl(path)
    return ([r["model_predictions"][0] for r in rows], [r["instruction"] for r in rows],
            [r["image_path"] for r in rows], [r["img_id"] for r in rows])


def assistant_answer(prediction, model):
    if model == "qwen":
        return prediction.split("assistant\n")[1]
    return prediction
