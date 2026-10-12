# HalluVL-DPO preference data

Preference pairs that teach a vision-language model to answer from the image rather than from what the prompt assumes. Each pair has an image, a query, a **chosen** answer that is grounded in the image and a **rejected** answer that hallucinates. Both answers are written by the model being trained (LLaVA-1.5-7B or Qwen2-VL-7B), so there is one dataset per model.

This folder has the pipeline that builds the data. The [repository README](../../README.md) summarizes the paper.

- [Use the released data](#use-the-released-data)
- [Set up the environment](#set-up-the-environment)
- [Build the data](#build-the-data)
- [How the pipeline works](#how-the-pipeline-works)
- [Differences between the paper, the released data and this code](#differences-between-the-paper-the-released-data-and-this-code)

## Use the released data

The datasets used in the paper are on Hugging Face: [PekhPekhPekh/HalluVL-DPO](https://huggingface.co/datasets/PekhPekhPekh/HalluVL-DPO).

| Config | Content |
|---|---|
| `qwen_mixture` | training set used for Qwen2-VL-7B (17,198 pairs) |
| `qwen_full` | every Qwen2-VL pair that passed the filters (105,000) |
| `llava_mixture` | training set used for LLaVA-1.5-7B (19,999 pairs) |
| `llava_full` | every LLaVA-1.5 pair that passed the filters (129,930) |
| `images` | the 27,471 images, stored once (`path`, `image`) |

One pair:

```json
{
  "id": 8056,
  "image": "8056.jpg",
  "type": "pope",
  "prompt": "<image>\nCan you find a gazebo in this image?",
  "chosen": "Yes, there is a gazebo in the image. It is a small, wooden structure with a curved roof and a window.",
  "rejected": "There is no gazebo in the image."
}
```

```python
from datasets import load_dataset

pairs = load_dataset("PekhPekhPekh/HalluVL-DPO", "qwen_mixture", split="train")
images = load_dataset("PekhPekhPekh/HalluVL-DPO", "images", split="train")
row_of = {path: i for i, path in enumerate(images["path"])}
image = images[row_of[pairs[0]["image"]]]["image"]   # PIL image
```

The pipeline below writes the same format.

## Set up the environment

One conda environment covers every step. It was tested with Python 3.10, PyTorch 2.3.1 (CUDA 12.1) and transformers 4.47.1.

```bash
conda env create -f environment.yml
conda activate halluvl
```

Grounding-DINO (step 2_2 only) is installed from its repository; the pins keep NumPy below 2, which spaCy needs. Its CUDA extension compiles against the CUDA toolkit pointed to by `CUDA_HOME`, which should match PyTorch's CUDA version (12.1); without it, Grounding-DINO falls back to a slower CPU implementation.

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO && pip install --no-build-isolation -e . "numpy<2" "opencv-python<4.11" && cd ..
```

API access:

```bash
export OPENAI_API_KEY=...   # GPT-5-mini: steps 1, 3_1 and 4_3
export HF_TOKEN=...         # Stable Diffusion 3 medium is a gated model on Hugging Face: step 2_1
```

Hardware: a GPU with at least 24 GB of memory for the 7B models and Stable Diffusion 3. Image generation and Grounding-DINO require a GPU; answer generation also runs on CPU, but slowly. All other models (Qwen2-VL-7B-Instruct, llava-hf/llava-1.5-7b-hf, BAAI/bge-m3, GPT-2, Grounding-DINO Swin-B) are downloaded from Hugging Face on first use.

## Build the data

You need:

- the CC3M training captions as a JSON list of `{"caption": ...}` records;
- the [concreteness ratings](https://doi.org/10.3758/s13428-013-0403-5) of Brysbaert et al. (2014) as a CSV with columns `Word` and `Conc.M`;
- the Grounding-DINO repository (for its Swin-B config file).

Edit [`scripts/config.sh`](scripts/config.sh) (data folder, input paths, number of captions, `MODEL=qwen` or `llava`, number of GPUs), then run:

```bash
bash scripts/01_images.sh     # captions, prompts for Stable Diffusion, images, object verification
bash scripts/02_prompts.sh    # queries with and without hints
bash scripts/03_answers.sh    # the target model answers every query; injection and attribute answers
bash scripts/04_pairs.sh      # chosen/rejected pairs, one pool per pair type
bash scripts/05_dataset.sh    # full dataset and training mixture
```

Steps 01 and 02 do not depend on the model; to build both datasets, run them once, then run 03 to 05 with `MODEL=qwen` and again with `MODEL=llava`.

Scale with the default 30,000 captions: about 6k GPT calls for captions and questions, one GPT call per TPQ (about 27k) for attribute answers, and 13 answer runs over all images per model. Answer generation is resumable: rerunning `03_answers.sh` continues where it stopped.

Outputs, under the data folder set in `config.sh`:

```
captions.json, triplets.jsonl, detections.jsonl, verified.jsonl
images/<id>.jpg                 one image per kept caption; id = its line in triplets.jsonl
prompts/                        queries, plain (text) and hinted (text_hint)
answers/<model>/                the model's answers
pairs/<model>/<type>.json       one pool of pairs per type
dataset/<model>_full.json       all pairs
dataset/<model>_mixture.json    training mixture
```

Every script also runs on its own; `python <script> --help` lists its arguments.

## How the pipeline works

**Images (01).** CC3M captions are kept when they parse into a single sentence (spaCy), read fluently (GPT-2 perplexity below 100) and name at least two concrete nouns (concreteness 4.5 or more), up to 30,000 captions. GPT-5-mini rewrites them in batches of 10 into a Stable Diffusion prompt with a list of present objects and a list of plausible absent objects. Stable Diffusion 3 medium draws each image, with the absent objects as negative prompt. Grounding-DINO (Swin-B) then keeps present objects it detects with confidence above 0.45 and absent objects it does not detect; images left without either are dropped.

**Queries (02).** For each image:

- GPT-5-mini writes a CPQ (assumes an absent object) and a TPQ (asks about a present object). Questions similar to earlier ones are passed back as examples to avoid, which keeps them diverse.
- POPE questions ask whether the first present object, or the first absent one, is in the image.
- A description request is drawn from seven templates.

Every query gets a hinted version (`text_hint`) next to the plain one (`text`).

**Answers (03).** The model answers both versions of every query with greedy decoding (`4_1_generate_answers.py`). Two more sets are derived from these answers:

- **Hallucination injection.** In the model's own description, one sentence is replaced by a phrase asserting an absent object ("The scene includes a dog"). Later sentences are more likely to be picked. The model then continues from there.
- **Answer inversion.** GPT-5-mini rewrites each TPQ answer with wrong attributes.

**Pairs (04).** For each pair type, one answer is chosen and one rejected:

| `type` | Chosen | Rejected | Threshold |
|---|---|---|---|
| `cpq` | CPQ with a true hint ("there is no X") | plain CPQ | 0.9 |
| `tpq` | plain TPQ | TPQ with a false hint ("there is no X") | 0.9 |
| `pope` | plain POPE, present object | POPE with "Hint: There is no X" | 0.9 |
| `pope_negative` | plain POPE, absent object | POPE with "Hint: There is a X" | 0.9 |
| `description_unilateral` | description with a true hint (caption, present and absent objects) | plain description | 0.9 |
| `description_contrastive` | description hinted with the present objects | description hinted with the absent objects | Qwen 0.92, LLaVA 0.9 |
| `description_injection` | plain description | injected description | 0.95 |
| `attribute` | plain TPQ answer | GPT's wrong-attribute answer | none |

A pair is dropped when its two answers are too similar: the cosine similarity of their BGE-M3 embeddings is above the threshold. For CPQ, TPQ and POPE it is also dropped when keywords show the hint did not change the answer (for example, a rejected CPQ answer that already says the object is absent); see `keep_pair` in `5_1_filter_pairs.py`.

**Dataset (05).** `<model>_full.json` holds every pair with its `type`. `<model>_mixture.json` samples the pools in the proportions 5000:5000:3000:3000:3000:3000:3000:5000 (CPQ, TPQ, POPE, POPE-negative, three description types, attribute), scaled to 20,000 pairs, with seed 42. A pool smaller than its share is used whole, so no pair repeats; the script prints the actual count per type.

| Script | Step |
|---|---|
| `0_caption_pool.py` | filter CC3M captions |
| `1_triplets.py` | GPT: SD prompt, present and absent objects |
| `2_1_generate_images.py` | Stable Diffusion 3 images |
| `2_2_grounding.py`, `2_3_filter_detections.py` | Grounding-DINO verification |
| `3_1_cpq_tpq_questions.py`, `3_2_cpq_tpq_prompts.py` | CPQ and TPQ queries |
| `3_3_pope_prompts.py` | POPE queries |
| `3_4_description_prompts.py` | description queries |
| `4_1_generate_answers.py` | model answers |
| `4_2_hall_inject_prompts.py` | hallucination-injection prompts |
| `4_3_attribute_answers.py` | GPT wrong-attribute answers |
| `5_1_filter_pairs.py`, `5_2_attribute_pairs.py` | pairs |
| `6_build_dataset.py` | full dataset and training mixture |
| `utils.py` | shared helpers |

## Differences between the paper, the released data and this code

The released data was built with an earlier version of this code, in two batches of 15,000 captions whose images were then numbered together. Known differences:

- **Model answers.** The released answers were generated with the authors' internal evaluation code. `4_1_generate_answers.py` re-implements it with Hugging Face transformers: same models, greedy decoding, the same answer formats and 128 new tokens for short queries. The description limit of the original runs was not recorded; this code uses 512 tokens. LLaVA here is the Hugging Face port `llava-hf/llava-1.5-7b-hf`.
- **Pair similarity.** The paper names Sentence-BERT; the code uses BGE-M3 embeddings (`[CLS]` token). The paper gives 0.9 as the threshold for all types except injection; the Qwen contrastive-hint pairs used 0.92.
- **Mixture size.** The released Qwen mixture has 17,198 pairs instead of 20k: its POPE-negative pool had only 1,745 pairs, so all shares were scaled down to avoid repeats. This code keeps the 20k target and uses small pools whole instead.
- **Weighted variant.** The pair weights of HalluVL-DPO^w are not part of this release.
- **POPE wording.** In released `pope` pairs with ids below 50,000, the question always uses "a" before the object, even for plurals, and is lower-case after the first letter. POPE and description templates were drawn without a fixed seed.
