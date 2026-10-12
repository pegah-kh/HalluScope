import argparse
from typing import Callable, Tuple

import torch

__all__ = ["get_model_class"]


SUPPORTED_MODELS = [
    "Qwen/Qwen3.5-9B",
]


def get_model_class(
    model_name_or_path: str = "Qwen/Qwen3.5-9B",
    model_base: str = "",
    processor_name: str = "Qwen/Qwen3.5-9B",
    device: torch.device = torch.device("cuda"),
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Tuple[Callable]:

    if "Qwen3.5" in model_name_or_path or "Qwen3.5" in model_base:
        from models.qwen3_5_vl import QwenVL

        model_class = QwenVL(
            model_name_or_path=model_name_or_path,
            model_base=model_base,
            processor_name=processor_name,
            local_files_only=args.local_files_only,
            cache_dir=args.cache_dir,
            message_format="role",
        )
        model_class.model_.to(device)

    else:
        raise NotImplementedError(
            f"Got {model_name_or_path}, but only these models are supported {SUPPORTED_MODELS}"
        )

    if logger is not None:
        logger.info(f"Successfully loaded {model_name_or_path}, device: {device}")
        logger.info(model_class)
        logger.info(model_class.model_)

    return model_class
