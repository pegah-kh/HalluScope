import argparse
import csv
import json

import spacy
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils import read_json


def load_concreteness(csv_path):
    scores = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                scores[row["Word"].lower()] = float(row["Conc.M"])
            except ValueError:
                continue
    return scores


class CaptionFilter:
    def __init__(self, concreteness, device, perplexity_threshold=100, min_concrete_nouns=2, concreteness_threshold=4.5):
        self.nlp = spacy.load("en_core_web_trf")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.lm = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
        self.device = device
        self.concreteness = concreteness
        self.perplexity_threshold = perplexity_threshold
        self.min_concrete_nouns = min_concrete_nouns
        self.concreteness_threshold = concreteness_threshold

    def perplexity(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            loss = self.lm(**inputs, labels=inputs["input_ids"]).loss
        return torch.exp(loss).item()

    def is_well_structured(self, sentence):
        doc = self.nlp(sentence)
        has_single_root = len([t for t in doc if t.dep_ == "ROOT"]) == 1
        return has_single_root and doc[-1].pos_ in {"PUNCT", "VERB", "NOUN", "ADJ"}

    def has_enough_concrete_nouns(self, sentence):
        count = sum(
            1 for token in self.nlp(sentence)
            if token.pos_ == "NOUN" and self.concreteness.get(token.lemma_.lower(), 0) >= self.concreteness_threshold
        )
        return count >= self.min_concrete_nouns

    def __call__(self, sentence):
        if len(sentence.strip().split()) < 3:
            return False
        if not self.is_well_structured(sentence):
            return False
        return self.has_enough_concrete_nouns(sentence) and self.perplexity(sentence) < self.perplexity_threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select fluent, well-formed CC3M captions with concrete objects.")
    parser.add_argument("--cc3m_file", required=True, help="CC3M annotations: JSON list of {'caption': ...}.")
    parser.add_argument("--concreteness_csv", required=True, help="Brysbaert et al. concreteness ratings (columns Word, Conc.M).")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--start_index", type=int, default=0, help="Index of the first CC3M caption to consider.")
    parser.add_argument("--target_count", type=int, default=30000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_valid = CaptionFilter(load_concreteness(args.concreteness_csv), device)
    samples = read_json(args.cc3m_file)[args.start_index:]

    pool = []
    for sample in tqdm(samples, desc="Filtering captions"):
        caption = sample["caption"].strip()
        if is_valid(caption):
            pool.append(caption)
            if len(pool) >= args.target_count:
                break

    with open(args.output_path, "w") as f:
        json.dump(pool, f, indent=2)
