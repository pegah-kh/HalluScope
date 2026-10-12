# HalluScope

**Benchmark construction pipeline for evaluating object hallucination in vision-language models.**

Starting from a diverse pool of COCO image captions, HalluScope annotates the objects present in each image, uses object co-occurrence statistics to pick a plausible but *absent* ("adversarial") object per image, and generates presupposition and presence ("POPE-style") questions that probe whether a model hallucinates that object.

**To evaluate a model, you only need [`benchmark/`](benchmark/README.md) and [Evaluation](#evaluation)**; the rest of this page describes how the benchmark is built.

---

## Benchmark versions

The final test sets are in [`benchmark/`](benchmark/README.md), also on Hugging Face as [PekhPekhPekh/HalluScope](https://huggingface.co/datasets/PekhPekhPekh/HalluScope), in two versions:

| Version | What it is |
|---|---|
| `benchmark/v1/` | The test set used for **all results in the paper**. Built with the earlier version of this pipeline, where the vision-language model used for captioning and for checking which objects are present or absent was **Qwen2-VL-7B-Instruct** (instead of Qwen3.5-9B in v2). |
| `benchmark/v2/` | Regenerated with the code in this repository (same method, cleaned-up pipeline). |

Scores on v1 and v2 are not directly comparable (different images and questions), so report which version you use. Each version has 3,000 COCO images per subset (`florence`, `instances`), each with four questions; see [`benchmark/README.md`](benchmark/README.md) for the question sets and the file format.

---

## Pipeline overview

```
[1] src/1_diverse_caption_pool     Select a diverse pool of COCO captions (K-Center Greedy)
                ↓
[2] src/2_object_annotation        Caption each image (Qwen3.5), detect objects (Florence-2 + COCO instances),
                                    then verify each object is really there (Grounding DINO + Qwen3.5)
                ↓
[3] src/3_adversary_object_detection   Pick, per image, an object that often co-occurs with its objects (PMI)
                                        but is verified absent, plus a random absent object; keep images
                                        with rich enough context (> 3 present objects)
                ↓
[4] src/4_question_generation      Generate adversary/description/POPE questions, then select a
                                    category-balanced diverse test set

src/evaluation                     Run a model on the benchmark and score its answers
```

Stages 2–4 are run **twice** — once for objects detected by Florence-2 (`florence`), once for COCO's own instance annotations (`instances`) — and the two subsets are carried through the whole pipeline side by side.

---

## Repository layout

```
halluscope/
├── benchmark/                  the released test sets (v1: paper, v2: regenerated)
├── environment.yml             main conda environment
├── environment_florence.yml    conda environment for Florence-2 detection only
├── data_files/                 generated at runtime — inputs and outputs of every stage live here
└── src/
    ├── 1_diverse_caption_pool/
    ├── 2_object_annotation/          (+ models/: the Qwen3.5 wrapper used by stages 2 and 3)
    ├── 3_adversary_object_detection/
    ├── 4_question_generation/
    └── evaluation/
```

Each stage directory has a `script.sh` that runs its steps in order, on the GPUs of the current machine. Stages 2 and 3 also have `slurm_*.sh` scripts to spread the GPU-heavy steps over several machines (see [Running on several GPUs or machines](#running-on-several-gpus-or-machines)).

All scripts can be run from any folder: each `script.sh` moves to `halluscope/` first, and every default data path (`data_files/...`) is resolved against `halluscope/`, not the current folder.

```bash
cd halluscope
bash src/1_diverse_caption_pool/script.sh
bash src/2_object_annotation/script.sh
bash src/3_adversary_object_detection/script.sh
bash src/4_question_generation/script.sh
```

---

## Before running

A few things are environment-specific and must be set before the pipeline will run:

- **COCO dataset** — scripts take `--coco_dir`, the COCO root folder containing `annotations/captions_train2017.json`, `annotations/instances_train2017.json` and `train2017/`. The default is our machine's `/data/common/coco`; set `COCO_DIR` at the top of each `script.sh` / `slurm_*.sh`.
- **Hugging Face cache** — stages 2 and 3 take `--cache_dir` (default `/data/khayatan/cache`); set `CACHE_DIR` at the top of each script. **Qwen3.5 is loaded offline** (`--local_files_only` defaults to `True`), so download it into that cache first:
  ```bash
  hf download Qwen/Qwen3.5-9B --cache-dir <your cache dir>
  ```
  Florence-2 (`microsoft/Florence-2-large`) and Grounding DINO (`IDEA-Research/grounding-dino-tiny`) are downloaded automatically on first use.
- **Python environments** — two are needed, because Florence-2's remote code does not work with `transformers` 5.x while Qwen3.5 needs it:
  ```bash
  conda env create -f environment.yml            # "halluscope": everything except Florence-2 (stages 1–4, evaluation)
  conda env create -f environment_florence.yml   # "halluscope-florence": Florence-2 detection only
  ```
  The files pin the versions the released pipeline was run with. The main environment needs `torch` ≥ 2.6: recent `transformers` refuses to load `.bin` checkpoints with older versions (CVE-2025-32434), which affects BGE-M3 in stage 4.

  The stage-2 and stage-3 scripts call each environment's Python directly; set `QWEN_ENV_DIR` / `FLORENCE_ENV_DIR` / `ENV_DIR` at their top to your environments' folders (e.g. `$(conda info --base)/envs/halluscope`).

  If you also have packages installed with `pip install --user` (in `~/.local`), Python finds those **before** the environment's own packages, which can break imports with version clashes. Run with `export PYTHONNOUSERSITE=1` to ignore them; the environments above contain every package the code needs.
- **LLM for stage 4 and the evaluation judge** — `generate_adversary_questions.py` and `evaluation/evaluate_presupposition.py` use `gpt-5-mini` through the OpenAI API (the paper's setup), with the key in the `OPENAI_API_KEY` environment variable. Both also accept any model served with an OpenAI-compatible API (e.g. vLLM) via `--api_base` and `--llm_model` / `--judge_model`.

---

## Stage 1 — Diverse caption pool

`src/1_diverse_caption_pool/create_diverse_caption_pool.py`

Embeds all COCO training captions (`all-MiniLM-L6-v2`) and runs K-Center Greedy selection on the GPU to pick a diverse subset (default 30,000 captions, `--seed 42`), then keeps one caption per image.

```bash
bash src/1_diverse_caption_pool/script.sh
```

Produces:
- `data_files/selected_coco_indices_30k.npy` — indices into the original COCO caption annotations
- `data_files/selected_coco_metadata.json` — `img_ids` / `ids` / `captions` for the selected pool (consumed by every later stage as `--filtered_metadata_path`)

---

## Stage 2 — Object annotation

`src/2_object_annotation/script.sh` runs four steps:

1. **`1_qwen_captioning.py`** — captions every selected image with Qwen3.5-9B (thinking disabled), asking for a short, plain-text description of every clearly visible object, without guesses or absent objects. The captions are later given to Qwen as context for the presence questions. → `data_files/qwen_captions.json` (JSON Lines, one `{"id", "prediction", "image"}` per image).
2. **`2_florence_objs_detection.py`** — runs Florence-2-large open-vocabulary object detection (`<OD>`, batched) → `data_files/coco_florence_objects.json` (`{"coco_florence_objects": {img_id: [objects]}}`; images with no detection are left out).
3. **`3_1_florence_grounding_dino_filtering.py`** — checks that each Florence object is really in the image. Florence labels are cleaned first: some Open Images names are simplified (e.g. `kitchen & dining room table` → `table`, `coffee cup` → `cup`), generation glitches are fixed (`loudoudspeakers` → `loudspeaker`) or dropped (`lampplant`), and singular/plural variants are merged (`sneaker` → `sneakers`); see `LABEL_RENAMES` and `LABEL_DROPS`. Then, for each object:
   - Grounding DINO (tiny, not trained on COCO) finds a box with score > 0.6 → **kept**;
   - DINO finds a box but is unsure → kept if Qwen3.5 answers yes to "{caption} Is there a/an {object} in this image?" on the full image, or else on DINO's box crop;
   - DINO finds no box → kept only if Qwen3.5 answers yes to a strict question (clearly visible, mostly in view, really that object and not a similar one or a picture of it), asked without the caption.
4. **`3_2_instances_grounding_dino_filtering.py`** — the same checks for the COCO instance categories of each image (each category is checked once per image).

Steps 3 and 4 produce:
- `data_files/florence_dino_filtered_objects.json` (`{"coco_florence_objects": {img_id: [kept]}, "rejected_objects": {img_id: [rejected]}}`)
- `data_files/instances_dino_filtered_objects.json` (`{"coco_instances_objects": {img_id: [kept]}, "rejected_objects": {img_id: [rejected]}}`)

Rejected objects are uncertain (detected or annotated, but not confirmed), so stage 3 never uses them as present **or** absent objects.

```bash
bash src/2_object_annotation/script.sh
```

---

## Stage 3 — Adversary object detection

`src/3_adversary_object_detection/script.sh` runs two steps, each for both `florence` and `instances`:

1. **`1_context_aware_adversary_discovery.py`** — computes PMI between objects from their co-occurrence over all images (adding 5 to each object's count, `--pmi_smoothing`, so that labels seen in only a few images do not dominate), and for each image with more than 3 present objects (`--min_objects`; the others are dropped in step 2 anyway):
   - ranks candidate objects by their total positive PMI with the image's present objects; an image's kept or rejected objects in **either** stage-2 file are never candidates;
   - takes as **absent object** the best-ranked of the top 10 candidates confirmed absent: Grounding DINO is not confident (score ≤ 0.6), Qwen3.5 answers no on the full image (asked even when DINO finds no box), and no on DINO's box crop when there is one;
   - draws a **random object** (seeded per image, `--seed`) with the same exclusions and the same absence check;
   - takes as **present object** the image's object most related (PMI) to the absent one.

   Writes one JSON-Lines record per image to `data_files/coco_{florence,instances}_adversary_random_present_objects.json`, with fields `img_id`, `img_filename`, `captions` (COCO ground-truth captions), `present_object`, `absent_object`, `random_object`, `all_present_objects`. Images where no absent or random object is confirmed are skipped. The per-GPU part files also log every candidate checked and which check rejected it; `--merge` prints a summary.
2. **`2_filtering_samples.py`** — keeps only images associated with **more than 3** present objects (richer context), producing `data_files/coco_{florence,instances}_adversary_random_present_objects_filtered.json` with an added `present` field (the full object list for that image).

```bash
bash src/3_adversary_object_detection/script.sh
```

---

## Running on several GPUs or machines

The GPU-heavy scripts — `1_qwen_captioning.py`, `2_florence_objs_detection.py`, `3_1_…`, `3_2_…` and `1_context_aware_adversary_discovery.py` — split their images over every GPU taking part: `accelerate launch --num_processes=N` splits over the GPUs of one machine, and `--shard_id i --num_shards M` over M separate jobs. Each GPU writes its own part file (`<output>_partXXXofYYY.json[l]`), resumes from it when restarted, and writes a `.done` marker when its whole slice is finished. Once all parts are done, they are merged into the usual output file:

- with `script.sh` (one machine), the merge runs automatically after each step;
- with the `slurm_*.sh` scripts (a SLURM array of 3 jobs × 2 GPUs, so 6 parts), merge by hand once every job has finished:

| Step | Submit | Merge |
|---|---|---|
| Captioning | `sbatch src/2_object_annotation/slurm_qwen_captioning.sh` | `cat data_files/qwen_captions_part*of006.json > data_files/qwen_captions.json` |
| Florence detection | `sbatch src/2_object_annotation/slurm_florence_detection.sh` | `python src/2_object_annotation/2_florence_objs_detection.py --merge --num_parts 6` |
| Florence verification (3_1) | `sbatch src/2_object_annotation/slurm_dino_filtering.sh` | `python src/2_object_annotation/3_1_florence_grounding_dino_filtering.py --merge --num_parts 6` |
| Instances verification (3_2) | `sbatch src/2_object_annotation/slurm_instances_filtering.sh` | `python src/2_object_annotation/3_2_instances_grounding_dino_filtering.py --merge --num_parts 6` |
| Adversary discovery | `sbatch src/3_adversary_object_detection/slurm_adversary_discovery.sh florence` (and `instances`) | `python src/3_adversary_object_detection/1_context_aware_adversary_discovery.py --objects_name coco_florence_objects --merge --num_parts 6` (and `coco_instances_objects`) |

Submit from `halluscope/`. The `--merge` commands refuse to merge while a part has no `.done` marker; resubmit the unfinished jobs (e.g. `sbatch --array=1,2 …`) with the same array size and GPU count, and they resume where they stopped. Keep `NUM_SHARDS` in each `slurm_*.sh` equal to the array size.

---

## Stage 4 — Question generation

`src/4_question_generation/script.sh` runs steps 1–3 for both `florence` and `instances` (`--subset`), then step 4:

1. **`generate_adversary_questions.py`** — asks `gpt-5-mini` to write, per sample, a presupposition question about the absent object (`adversary_question`) and a grounded question about the present object (`truth_question`), given the first COCO caption. To keep questions varied, each request lists questions already written for the most similar captions (BGE-M3 embeddings) as questions to avoid. Samples go 20 per request; question pairs are matched to samples by position, so a request is retried if the LLM returns a different number of pairs. Progress is saved as it goes, so a rerun only generates what is missing (e.g. after failed requests) → `data_files/coco_{florence,instances}_difficult_samples_adversary_questions.json`, in input order.
2. **`generate_description_questions.py`** — attaches a randomly chosen (seeded) open-ended "describe this image" prompt → `data_files/{florence,instances}_description_questions.json`.
3. **`generate_pope_questions.py`** — generates three POPE-style yes/no presence questions per image from 7 templates (seeded; with the right article, and "are" for plural objects): about the present object, the contextual absent object, and the random absent object → `data_files/{florence,instances}_pope_questions.json` (+ `_negative.json`, `_random.json`).
4. **`select_test_set.py`** — describes each adversary question by its intent (color, material, position, … from keyword patterns) and its absent object, embeds both, and runs K-Center Greedy selection to pick a diverse, category-balanced set of images (default 3,000 per subset); the four question sets of these images form the test set → `data_files/test_dataset/{adversary_presupposition,real_recognition,adversary_recognition,random_recognition}_{florence,instances}.json`.

```bash
bash src/4_question_generation/script.sh
```

These four files per subset (8 total) are the final HalluScope benchmark.

---

## Evaluation

`src/evaluation/` runs a vision-language model on the test sets and scores its answers. It is separate from building the benchmark: it only needs `benchmark/` (choose the version with `--version v1|v2`, default v2) and the COCO `train2017/` images.

The test sets, each for the `florence` and `instances` subsets (`{task}_{subset}.json`):

| Task | Question about | Expected answer |
|---|---|---|
| `adversary_presupposition` | the absent, context-related object, presupposing it is there ("What color is the bow and arrow?") | not asserting the object |
| `real_recognition` | the present object ("Is there a …?") | yes |
| `adversary_recognition` | the absent, context-related object | no |
| `random_recognition` | an absent, random object | no |

The three steps (all in `src/evaluation/script.sh`):

1. **`run_inference.py`** — answers every question with any Hugging Face chat VLM that loads with `AutoModelForImageTextToText` (e.g. Qwen2.5-VL, Qwen3.5, `llava-hf/llava-1.5-7b-hf`), greedily, on one GPU. Writes `data_files/predictions/<run_name>/{task}_{subset}.jsonl`: each test item plus `"prediction"`. Resumes an interrupted run; `--limit N` answers only the first N questions of each test set, for a quick check. To evaluate your own model without this script, write these files yourself.
2. **`evaluate_recognition.py`** — the answer is the first whole word "yes" or "no"; answers with neither (e.g. "I don't know") count as wrong and are reported as invalid. Reports per test set the accuracy and share of "yes" answers, and POPE-style accuracy / precision / recall / F1 on real + adversary ("adversarial") and real + random ("random"), per subset and overall. Writes `data_files/scores/<run_name>/recognition.json`.
3. **`evaluate_presupposition.py`** — a judge LLM decides whether each answer asserts the absent object is present; accuracy is the share that does not. The judge is `gpt-5-mini` by default (the paper's setup; set `OPENAI_API_KEY`), or any model behind an OpenAI-compatible server such as vLLM (`--api_base http://localhost:8000/v1 --judge_model <model>`). Judgments are saved as they come, so a rerun only judges what is missing. Writes `data_files/scores/<run_name>/presupposition.json`.

```bash
python src/evaluation/run_inference.py --model Qwen/Qwen2.5-VL-7B-Instruct --version v2 --coco_dir /path/to/coco   # -> run name Qwen2.5-VL-7B-Instruct_v2
python src/evaluation/evaluate_recognition.py --run_name Qwen2.5-VL-7B-Instruct_v2
OPENAI_API_KEY=... python src/evaluation/evaluate_presupposition.py --run_name Qwen2.5-VL-7B-Instruct_v2
```

`run_inference.py --test_dir data_files/test_dataset` evaluates on a test set you built yourself with stages 1–4 instead.

Evaluation runs in the main environment (`environment.yml`).

---

## Naming conventions

- Every `--save_filename`/`--save_dir` pair produced by these scripts is joined as `save_dir/save_filename + ".json"` — pass the file name **without** the `.json` suffix.
- Most `*.json` files here are actually **JSON Lines** (one JSON object per line), not a single JSON document — the exceptions are the object-annotation dictionaries (`{"coco_florence_objects": {...}}` etc.), `selected_coco_metadata.json`, and the final `data_files/test_dataset/*.json` files, which are single JSON arrays/objects.
- Per-GPU part files (`*_partXXXofYYY.json` / `.jsonl`) and their `.done` markers are intermediate; the merged file without the `_part` suffix is the output of each step.
- `florence` and `instances` refer to the two parallel object sources (Florence-2 open-vocabulary detections vs. COCO's own instance annotations) carried through every stage.
