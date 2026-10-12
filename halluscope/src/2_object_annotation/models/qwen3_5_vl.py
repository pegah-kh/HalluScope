from typing import Any, Callable, Dict, List, Union
import os

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from .image_text_model import ImageTextModel

__all__ = ["QwenVL"]




def get_local_model_path(cache_dir, model_name):
    """
    Find the correct local path for a model inside the cache directory.

    Args:
        cache_dir (str): Base cache directory (e.g., "/data/khayatan/cache/")
        model_name (str): Model name (e.g., "Qwen/Qwen3.5-9B")

    Returns:
        str: Path to the model's snapshot directory.
    """
    model_cache_dir = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")

    if not os.path.exists(model_cache_dir):
        raise FileNotFoundError(f"Model cache directory not found: {model_cache_dir}")

    snapshots_dir = os.path.join(model_cache_dir, "snapshots")
    if not os.path.exists(snapshots_dir):
        raise FileNotFoundError(f"Snapshots directory not found in: {model_cache_dir}")

    # Find the first snapshot hash folder
    snapshots = sorted(os.listdir(snapshots_dir), reverse=True)  # Sort in case of multiple versions
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in: {snapshots_dir}")

    return os.path.join(snapshots_dir, snapshots[0])  # Use the first snapshot found




class QwenVL(ImageTextModel):

    def set_model(
        self,
        cache_dir:str = None,
    ) -> None:

        if self.local_files_only:
            local_model_path = get_local_model_path(cache_dir, self.model_name_or_path if len(self.model_base)<=1 else self.model_base)
            self.local_model_path = local_model_path
        else:
            self.local_model_path = self.model_name_or_path if len(self.model_base)<=1 else self.model_base


        kwargs = {
            "dtype": torch.bfloat16,
            "low_cpu_mem_usage":True,
            "local_files_only":self.local_files_only,
            "cache_dir":cache_dir,
        }

        self.model_ = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.local_model_path,
            **kwargs
        )


    def get_model(
        self,
    ) -> Callable:

        return self

    def get_language_model(
        self,
    ) -> Callable:

        return self.model_.model

    def get_lm_head(
        self,
    ) -> Callable:

        return self.model_.lm_head

    def set_processor(
        self,
    ) -> None:

        self.processor_ = AutoProcessor.from_pretrained(
            self.local_model_path, # self.processor_name
            local_files_only=self.local_files_only,
            cache_dir=self.cache_dir,
            use_fast=True,
        )
        self.tokenizer_ = self.processor_.tokenizer

    def set_preprocessor(
        self,
    ) -> None:

        self.preprocessor_ = self.preprocess_input

    def get_conversation_round(
        self,
        instruction: str = "What are these?",
        response: str = "",
        image_file: str = "",
    ) -> List[Dict[str, Any]]:

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_file,
                    },
                    {"type": "text", "text": instruction},
                ],
            },
        ]
        if response:
            conversation.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": response},
                    ],
                },
            )

        return conversation

    def get_conversation_template(
        self,
        instruction: str = "What are these?",
        response: str = "",
        image_file: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:

        conversation = self.get_conversation_round(
            instruction=instruction,
            response=response,
            image_file=image_file,
        )
        return conversation

    def preprocess_text(
        self,
        conversation,
        generation_mode: bool = False,
        continue_final_message: bool = False,
        **kwargs: Any,
    ) -> str:
        add_generation_prompt = generation_mode if not continue_final_message else False


        # Qwen3.5 thinks by default; disable it so short answers (yes/no, captions) come back directly
        prompt = self.processor_.apply_chat_template(
            conversation,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
            enable_thinking=False,
        )

        return prompt

    def preprocess_images(
        self,
        conversation,
        **kwargs: Any,
    ) -> List:

        image_inputs, video_inputs = process_vision_info(conversation)

        return image_inputs, video_inputs


    def preprocess_input(
        self,
        instruction: Union[str, list[str]] = "What are these?",
        image_file: Union[str, list[str], None] = None,
        response: Union[str, list[str]] = "",
        generation_mode: bool = False,
        continue_final_message: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Accepts either a single sample or a batch of samples.
        Automatically detects type and returns properly batched inputs.
        """

        if isinstance(instruction, str):
            instruction = [instruction]
        if isinstance(image_file, str) or image_file is None:
            image_file = [image_file]
        if isinstance(response, str):
            response = [response]

        batch_size = len(instruction)

        conversations = [
            self.get_conversation_template(
                instruction=instruction[i],
                response=response[i] if i < len(response) else "",
                image_file=image_file[i] if i < len(image_file) else None,
            )
            for i in range(batch_size)
        ]
        imgs, vids = self.preprocess_images(conversations)


        all_texts = [
            self.preprocess_text(
                conv,
                generation_mode=generation_mode,
                continue_final_message=continue_final_message,
            )
            for conv in conversations
        ]

        inputs = self.processor_(
            text=all_texts,
            images=imgs if imgs else None,
            videos=vids if vids else None,
            padding=True,
            return_tensors="pt",
        )

        return inputs



    def generate(
        self,
        **kwargs: Any
    ):

        return self.model_.generate(**kwargs)
