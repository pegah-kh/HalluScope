---
license: cc-by-4.0
language:
  - en
task_categories:
  - visual-question-answering
pretty_name: HalluScope
configs:
  - config_name: v2_florence
    default: true
    data_files:
      - {split: adversary_presupposition, path: v2/adversary_presupposition_florence.json}
      - {split: real_recognition, path: v2/real_recognition_florence.json}
      - {split: adversary_recognition, path: v2/adversary_recognition_florence.json}
      - {split: random_recognition, path: v2/random_recognition_florence.json}
  - config_name: v2_instances
    data_files:
      - {split: adversary_presupposition, path: v2/adversary_presupposition_instances.json}
      - {split: real_recognition, path: v2/real_recognition_instances.json}
      - {split: adversary_recognition, path: v2/adversary_recognition_instances.json}
      - {split: random_recognition, path: v2/random_recognition_instances.json}
  - config_name: v1_florence
    data_files:
      - {split: adversary_presupposition, path: v1/adversary_presupposition_florence.json}
      - {split: real_recognition, path: v1/real_recognition_florence.json}
      - {split: adversary_recognition, path: v1/adversary_recognition_florence.json}
      - {split: random_recognition, path: v1/random_recognition_florence.json}
  - config_name: v1_instances
    data_files:
      - {split: adversary_presupposition, path: v1/adversary_presupposition_instances.json}
      - {split: real_recognition, path: v1/real_recognition_instances.json}
      - {split: adversary_recognition, path: v1/adversary_recognition_instances.json}
      - {split: random_recognition, path: v1/random_recognition_instances.json}
---

![HalluScope](assets/halluscope_logo.png)

# HalluScope benchmark

HalluScope tests *why* a vision-language model hallucinates objects. Each image comes with four questions:

| Question set | Asks about | Example | Correct answer |
|---|---|---|---|
| `real_recognition` | an object that **is** in the image | "Is there a baseball bat in the image?" | yes |
| `random_recognition` | an absent object, unrelated to the scene | "Is there a statue in the image?" | no |
| `adversary_recognition` | an absent object that usually appears in such scenes | "Does the image contain a baseball cap?" | no |
| `adversary_presupposition` | the same absent object, in a question that assumes it is there | "What logo appears on the baseball cap worn by the batter?" | an answer that does not claim the object exists |

Comparing the four tells apart the causes: a model failing only the last one is misled by the **question**, not by what it sees.

Images are from COCO `train2017` (not included; use the `image` file name). There are two subsets of 3,000 images each, with the same images across the four question sets of a subset:
- `florence`: objects detected by Florence-2, then verified;
- `instances`: COCO's own object annotations, then verified.

## Versions

| Version | What it is |
|---|---|
| **v1** | The test set used for **all results in the paper**. Built with the earlier version of the pipeline, where the vision-language model used for captioning and for checking which objects are present or absent was **Qwen2-VL-7B-Instruct** (instead of Qwen3.5-9B in v2). |
| **v2** | Regenerated with the released, cleaned-up pipeline (same method). Use it to compare with models evaluated on v2, or to reproduce the construction end to end. |

Scores on v1 and v2 are **not** directly comparable: the images and questions differ. Report which version you use.

What changed in v2: captions and verification with Qwen3.5-9B; Florence label cleanup (broken or duplicate labels fixed or removed); a stricter check before an object is considered present or absent; smoothed co-occurrence statistics so that rare labels are no longer over-chosen as absent objects; fixes in question generation (questions can no longer be matched to the wrong image, plural-aware yes/no questions).

## Format

Each file is a JSON array; each item is, for example (v2, `adversary_presupposition`, `instances`; the image shows a slice of cake with a fork, and no knife):

<img src="https://s3.amazonaws.com/images.cocodataset.org/train2017/000000448600.jpg" alt="COCO image 000000448600: a slice of cake on a plate, with a fork" width="280">

<sub>COCO image 000000448600 (from Flickr), licensed under <a href="http://creativecommons.org/licenses/by-nc-sa/2.0/">CC BY-NC-SA 2.0</a>.</sub>

```json
{
  "image": "000000448600.jpg",
  "present_object": "fork",
  "absent_object": "knife",
  "random_object": "toilet",
  "text": "What material is the knife beside the plate made of?"
}
```

`text` is the question; `present_object`, `absent_object` and `random_object` say which object each question set is about.

## Loading

```python
from datasets import load_dataset

questions = load_dataset("PekhPekhPekh/HalluScope", "v2_florence", split="adversary_presupposition")
```

Or use the JSON files directly with the evaluation scripts of the [HalluScope code](https://github.com/pegah-kh/halluscope) (`src/evaluation`, `--version v1|v2`).

## Citation

```bibtex
@inproceedings{khayatan2026prompts,
  title     = {When Prompts Override Vision: Prompt-Induced Hallucinations in {LVLMs}},
  author    = {Khayatan, Pegah and Parekh, Jayneel and Dapogny, Arnaud and Shukor, Mustafa and Newson, Alasdair and Cord, Matthieu},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
