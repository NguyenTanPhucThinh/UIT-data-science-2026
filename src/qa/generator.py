from __future__ import annotations

import logging
from typing import Any, Sequence

from .config import LegalQAGenerationConfig, LLMConfig

logger = logging.getLogger(__name__)


class LocalLLMGenerator:
    """Load and run a Hugging Face causal LLM entirely on local hardware."""

    def __init__(self, model_config: LLMConfig):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Task 2 requires transformers, torch, accelerate and safetensors"
            ) from exc

        self.torch = torch
        self.model_config = model_config
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name_or_path,
            trust_remote_code=model_config.trust_remote_code,
            local_files_only=model_config.local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        dtype = self._resolve_dtype(model_config.dtype)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": model_config.trust_remote_code,
            "local_files_only": model_config.local_files_only,
            "low_cpu_mem_usage": True,
        }
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        if model_config.device == "auto":
            model_kwargs["device_map"] = "auto"

        logger.info("Loading local LegalQA LLM: %s", model_config.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            **model_kwargs,
        )
        if model_config.device != "auto":
            self.model.to(model_config.device)
        # The published Vi-Qwen2-1.5B-RAG config has use_cache=false.  That is
        # appropriate for training, but makes autoregressive inference
        # repeatedly recompute the whole prompt and can exhaust an 8 GiB GPU.
        self.model.config.use_cache = True
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.use_cache = True
        self.model.eval()

    def _resolve_dtype(self, dtype: str):
        if dtype == "float32":
            return self.torch.float32
        if dtype == "float16":
            return self.torch.float16
        if dtype == "bfloat16":
            return self.torch.bfloat16
        if dtype != "auto":
            raise ValueError("dtype must be auto, float32, float16, or bfloat16")
        if not self.torch.cuda.is_available():
            return self.torch.float32
        return self.torch.bfloat16 if self.torch.cuda.is_bf16_supported() else self.torch.float16

    @property
    def input_device(self):
        return next(self.model.parameters()).device

    def count_message_tokens(self, messages: Sequence[dict[str, str]]) -> int:
        encoded = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(encoded)

    def generate_batch(
        self,
        messages_batch: Sequence[Sequence[dict[str, str]]],
        generation_config: LegalQAGenerationConfig,
    ) -> list[str]:
        if not messages_batch:
            return []
        rendered = [
            self.tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
            for messages in messages_batch
        ]
        inputs = self.tokenizer(rendered, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.input_device) for key, value in inputs.items()}
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": generation_config.max_new_tokens,
            "do_sample": generation_config.do_sample,
            "use_cache": True,
            "repetition_penalty": generation_config.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if generation_config.do_sample:
            generate_kwargs.update(
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
            )
        else:
            # Override sampling-only values shipped in the model's
            # generation_config.json so greedy decoding stays warning-free.
            generate_kwargs.update(temperature=None, top_p=None, top_k=None)
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        prompt_width = inputs["input_ids"].shape[1]
        return self.tokenizer.batch_decode(
            outputs[:, prompt_width:], skip_special_tokens=True
        )
