<p align="center">
  <img src="docs/images/halluscope_logo.png" alt="HalluScope" width="600">
</p>

# When Prompts Override Vision: Prompt-Induced Hallucinations in LVLMs

[![arXiv](https://img.shields.io/badge/arXiv-2604.21911-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.21911)
[![Project page](https://img.shields.io/badge/Project-page-0a66c2?logo=googlechrome&logoColor=white)](https://pegah-kh.github.io/projects/prompts-override-vision/)
![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-6f42c1)

Pegah Khayatan, Jayneel Parekh, Arnaud Dapogny, Mustafa Shukor, Alasdair Newson, Matthieu Cord
(ISIR, Sorbonne Université · Valeo.ai)

The paper has two contributions, and this repository has one part for each:

| | What it is | Released | Code |
|---|---|---|---|
| [**HalluScope**](#halluscope) | A benchmark that separates the causes of hallucination in large vision-language models (LVLMs) | Test sets in [`halluscope/benchmark/`](halluscope/benchmark/README.md) | [`halluscope/`](halluscope/README.md): builds the benchmark and evaluates a model on it |
| [**HalluVL-DPO**](#halluvl-dpo) | A preference-optimization framework and dataset that reduces hallucinations caused by the text prompt | Preference datasets on [Hugging Face](https://huggingface.co/datasets/PekhPekhPekh/HalluVL-DPO) | [`halluvl/dpo/`](halluvl/dpo/README.md): the pipeline that builds the preference data |

## The problem

LVLMs often describe things that are not in the image. Usual benchmarks (POPE, CHAIR, MMHal-Bench) count these errors but cannot tell where they come from:

1. **Perception:** the vision encoder missing something.
2. **Co-occurrence priors:** the model's knowledge of which objects usually appear together.
3. **The prompt:** the prompt itself assuming something is there.

## HalluScope

HalluScope asks four kinds of questions about COCO images, each probing one cause:

| Question type | Example | Correct answer | What it tests |
|---|---|---|---|
| Positive recognition | "Is there a baseball bat in the image?" (present) | yes | visual perception |
| Random recognition | "Is there a statue in the image?" (absent, unrelated) | no | visual perception |
| Adversarial recognition | "Does the image contain a baseball cap?" (absent, but typical of the scene) | no | reliance on object co-occurrence |
| Adversarial presupposition (AdP) | "What logo appears on the baseball cap worn by the batter?" | not asserting the cap | reliance on what the prompt assumes |

Adversarial objects are chosen with co-occurrence statistics (pointwise mutual information) over verified object labels, and their absence is checked with Grounding-DINO plus a VLM. The benchmark has 3k images in each of two subsets (objects from COCO instances or from Florence-2).

**Finding.** Across InternVL3.5-8B, Molmo-7B-D, Qwen2-VL-7B, Qwen2.5-VL-7B and LLaVA-1.5-7B, positive and random recognition stay above 85%: perception is mostly fine. Accuracy drops by about 8–37 points on adversarial objects, and by 25–85 points when the question presupposes the object. **The text prompt is the strongest driver of hallucination**, more than co-occurrence priors or the vision backbone.

**Get the benchmark:** [`halluscope/benchmark/`](halluscope/benchmark/README.md) has two versions: **v1**, used for all results in the paper, and **v2**, regenerated with the released code. Images are from COCO `train2017`.

**Code:** [`halluscope/`](halluscope/README.md) has the evaluation scripts (run any Hugging Face vision-language model on the benchmark and score it) and the full pipeline that builds the benchmark.

## HalluVL-DPO

HalluVL-DPO fine-tunes a model with DPO to prefer grounded answers over hallucinated ones. The target model itself writes both answers, which keeps the data close to its own distribution. Queries are presupposition questions about present (TPQ) or absent (CPQ) objects, POPE-style yes/no questions and detailed descriptions. The final set has about 20k pairs in a CPQ:TPQ:POPE:description:attribute ratio of 5:5:6:9:5. Gains carry over to CP-Bench, CHAIR and HallusionBench, while POPE, MME, ScienceQA and MM-Vet stay close to the original models.

The training data is built from 27.4k synthetic images (Stable Diffusion 3 from filtered CC3M captions, objects verified with Grounding-DINO). Example pairs, one per type:

<table>
  <tr><th>Image</th><th>Pair type</th><th>Question</th><th>Preferred (grounded)</th><th>Rejected (hallucinated)</th></tr>
  <tr>
    <td><img src="docs/images/cpq.jpg" width="140"></td>
    <td><b>CPQ</b>: presupposition question about an absent object</td>
    <td>What color is the spatula resting near the skillet?</td>
    <td>There is no spatula in the image.</td>
    <td>The spatula resting near the skillet is black.</td>
  </tr>
  <tr>
    <td><img src="docs/images/tpq.jpg" width="140"></td>
    <td><b>TPQ</b>: presupposition question about a present object</td>
    <td>What color is the bird?</td>
    <td>It has a black head with a white stripe, a yellow body with orange and red accents, and a black beak.</td>
    <td>The bird is not shown in the image.</td>
  </tr>
  <tr>
    <td><img src="docs/images/pope.jpg" width="140"></td>
    <td><b>POPE</b>: yes/no question, present object</td>
    <td>Do you see a elf on the shelf in the picture?</td>
    <td>Yes, there is an elf on the shelf in the picture. The elf is sitting next to a snowman.</td>
    <td>No, there is no elf on the shelf in the picture.</td>
  </tr>
  <tr>
    <td><img src="docs/images/pope_negative.jpg" width="140"></td>
    <td><b>POPE</b>: yes/no question, absent object</td>
    <td>Would you say there's a ladder visible?</td>
    <td>There is no ladder visible in the image.</td>
    <td>Yes, there is a ladder visible in the image. It is located on the side of the building.</td>
  </tr>
  <tr>
    <td><img src="docs/images/attribute.jpg" width="140"></td>
    <td><b>Model-assisted answer inversion</b> (attribute): an LLM rewrites a correct TPQ answer with wrong attributes</td>
    <td>What is the condition of the road surface?</td>
    <td>The road surface appears to be wet, likely due to recent or ongoing rain.</td>
    <td>The road surface appears dry and dusty, with a matte finish and no visible reflections or puddles.</td>
  </tr>
  <tr>
    <td><img src="docs/images/description_unilateral.jpg" width="140"></td>
    <td><b>Unilateral hint</b> (description): a true or false hint about an object is added to the query to obtain a grounded or a hallucinated answer</td>
    <td>Could you offer a thorough description of the image?</td>
    <td>The image features a man playing a <b>trumpet</b> in a clean studio setting. …</td>
    <td>The image features a man playing a <b>trombone</b>, with the instrument held in his hands. …</td>
  </tr>
  <tr>
    <td><img src="docs/images/description_contrastive.jpg" width="140"></td>
    <td><b>Contrastive hint</b> (description): opposite hints for both answers, which avoids the length bias the unilateral hint causes on descriptions</td>
    <td>Provide a detailed description of the image?</td>
    <td>The image features a dense forest with a large number of trees, including a couple of very tall trees. …</td>
    <td>The image features a lush green forest with a <b>trail</b> running through it. Along the trail, there is a <b>bench</b> … A <b>stream</b> can be seen nearby. …</td>
  </tr>
  <tr>
    <td><img src="docs/images/description_injection.jpg" width="140"></td>
    <td><b>Post-hoc hallucination injection</b> (description): one sentence of the model's description, more likely a later one, is replaced by a phrase asserting an absent object, and the model completes the rest</td>
    <td>Can you describe what you see in the image?</td>
    <td>The image depicts a view looking up through the canopy of tall trees … The sky is visible through the gaps in the leaves. …</td>
    <td>The image depicts a view looking up through the canopy of tall trees … The scene includes a <b>campfire</b>, which is positioned in the lower left corner of the image. …</td>
  </tr>
</table>

Only pairs whose answers are far apart in embedding space are kept. The paper also proposes a weighted variant (HalluVL-DPO<sup>w</sup>) where an LLM scores each pair's semantic gap from 1 to 3 to weight it in the loss.

**Get the data:** the preference datasets used in the paper, for LLaVA-1.5-7B and Qwen2-VL-7B, are on Hugging Face: [PekhPekhPekh/HalluVL-DPO](https://huggingface.co/datasets/PekhPekhPekh/HalluVL-DPO).

**Code:** [`halluvl/dpo/`](halluvl/dpo/README.md) has the pipeline that builds the preference data, from captions to images, questions, answer pairs and the final mixture. It does not contain the data itself.

## Repository

```
├── halluscope/
│   ├── benchmark/        HalluScope test sets (v1: paper, v2: regenerated)
│   ├── src/              pipeline that builds the benchmark, and evaluation scripts
│   └── README.md         how to evaluate a model and how the benchmark is built
├── halluvl/
│   └── dpo/              pipeline that builds the HalluVL-DPO preference data (data on Hugging Face)
└── docs/images/          figures used in this README
```

## Citation

```bibtex
@inproceedings{khayatan2026prompts,
  title     = {When Prompts Override Vision: Prompt-Induced Hallucinations in {LVLMs}},
  author    = {Khayatan, Pegah and Parekh, Jayneel and Dapogny, Arnaud and Shukor, Mustafa and Newson, Alasdair and Cord, Matthieu},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE).
