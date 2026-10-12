import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm

from utils import read_jsonl

DEFAULT_MODEL_PATHS = {"qwen": "Qwen/Qwen2-VL-7B-Instruct", "llava": "llava-hf/llava-1.5-7b-hf"}


class Qwen2VL:
    def __init__(self, model_path):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto").eval()

    def answer(self, image_path, text, max_new_tokens, continue_text=False):
        """Returns the full decoded conversation ("system\\n...user\\n...assistant\\n<answer>").

        With continue_text, `text` is continued as is, without opening an assistant turn: this is how the
        hallucination-injection prompts (question + "assistant\\n" + partial answer) were completed.
        """
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=not continue_text, continue_final_message=continue_text)
        images, _ = process_vision_info(messages)
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(output, skip_special_tokens=True)[0]


class LLaVA:
    def __init__(self, model_path):
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto").eval()

    def answer(self, image_path, text, max_new_tokens, prefix=""):
        """Returns the answer only; with a prefix, the answer starts with it."""
        prompt = f"USER: <image>\n{text} ASSISTANT:" + (f" {prefix}" if prefix else "")
        inputs = self.processor(images=Image.open(image_path).convert("RGB"), text=prompt,
                                return_tensors="pt").to(self.model.device, torch.bfloat16)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT:", 1)[1].strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer a prompt list with the target model (greedy decoding).")
    parser.add_argument("--model", choices=["qwen", "llava"], required=True)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--prompts", required=True, help="Prompt list from step 3 (jsonl with image, text, text_hint).")
    parser.add_argument("--field", choices=["text", "text_hint"], required=True)
    parser.add_argument("--inject", action="store_true",
                        help="Prompt list of 4_2_hall_inject_prompts.py: continue the injected partial answer.")
    parser.add_argument("--image_root", required=True, help="Folder that the prompt list's image paths are relative to.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_path", required=True, help="JSON lines read by the later steps.")
    args = parser.parse_args()

    model = (Qwen2VL if args.model == "qwen" else LLaVA)(args.model_path or DEFAULT_MODEL_PATHS[args.model])

    done = set()
    if os.path.exists(args.output_path):
        done = {row["image_path"] for row in read_jsonl(args.output_path)}

    with open(args.output_path, "a", encoding="utf-8") as f:
        for item in tqdm(read_jsonl(args.prompts)):
            if item["image"] in done:
                continue
            image_path = os.path.join(args.image_root, item["image"])
            if args.inject and args.model == "qwen":
                prediction = model.answer(image_path, item["text_hint"], args.max_new_tokens, continue_text=True)
            elif args.inject:
                prediction = model.answer(image_path, item["text"], args.max_new_tokens, prefix=item["text_hint"])
            else:
                prediction = model.answer(image_path, item[args.field], args.max_new_tokens)
            row = {
                "img_id": os.path.splitext(item["image"])[0],
                "image_path": item["image"],
                "instruction": item["text"] if args.inject else item[args.field],
                "model_predictions": [prediction],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
