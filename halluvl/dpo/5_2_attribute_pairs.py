import argparse

from utils import read_jsonl, write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn the answers of 4_3_attribute_answers.py into DPO pairs.")
    parser.add_argument("--attribute_answers", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    pairs = []
    for item in read_jsonl(args.attribute_answers):
        question = item["attribute_question"]
        pairs.append({"id": int(item["id"]), "image": item["image"],
                      "prompt": f"<image>\n{question[0].upper() + question[1:]}",
                      "chosen": item["right_answer"], "rejected": item["wrong_answer"]})
    write_json(pairs, args.output_path)
