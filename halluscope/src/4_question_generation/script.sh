#!/bin/bash
# Can be run from anywhere: moves to the project root (halluscope/halluscope/) first.
# Step 1 needs an LLM: OpenAI (the paper's setup, gpt-5-mini; set OPENAI_API_KEY) or an OpenAI-compatible
# server given with --api_base. Step 1 resumes where it stopped when rerun.
set -e
cd "$(dirname "$0")/../.."


# FOR FLORENCE OBJECTS (3_1) AND INSTANCES OBJECTS (3_2) SEPARATELY:
for SUBSET in florence instances; do

    # 1. ADVERSARY (PRESUPPOSITION) AND TRUTH QUESTIONS, WRITTEN BY AN LLM.
    # Writes data_files/coco_${SUBSET}_difficult_samples_adversary_questions.json
    python src/4_question_generation/generate_adversary_questions.py --subset $SUBSET

    # 2. DESCRIPTION QUESTIONS.
    # Writes data_files/${SUBSET}_description_questions.json
    python src/4_question_generation/generate_description_questions.py --subset $SUBSET

    # 3. POPE (PRESENCE) QUESTIONS: PRESENT, ADVERSARY AND RANDOM OBJECT.
    # Writes data_files/${SUBSET}_pope_questions{,_negative,_random}.json
    python src/4_question_generation/generate_pope_questions.py --subset $SUBSET

done


# 4. SELECT A CATEGORY-BALANCED, DIVERSE TEST SET (3,000 IMAGES PER SUBSET).
# Writes data_files/test_dataset/{adversary_presupposition,real_recognition,adversary_recognition,random_recognition}_{florence,instances}_balanced_w_category.json
python src/4_question_generation/select_balanced_set_samples_category.py --subsets florence instances
