"""Remote-only lazy model loading."""

from __future__ import annotations

import os
from typing import Any


def load_model(
    handle: str,
    *,
    dry_run: bool,
    remote: bool | None = None,
    dtype: str = "bfloat16",
) -> tuple[Any, Any] | None:
    """Load only when explicitly remote and not a dry-run."""
    is_remote = (
        remote if remote is not None else os.environ.get("KVCACHE_REMOTE") == "1"
    )
    if dry_run:
        return None
    if not is_remote:
        raise RuntimeError("model loading is remote-only; set KVCACHE_REMOTE=1")
    if not os.environ.get("KVCACHE_ALLOW_DOWNLOAD"):
        raise RuntimeError(
            "model access requires KVCACHE_ALLOW_DOWNLOAD=1 "
            "after confirming remote caches"
        )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for non-dry model execution")
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(handle, local_files_only=False)
    model = AutoModelForCausalLM.from_pretrained(
        handle,
        torch_dtype=torch_dtype,
        device_map="auto",
        local_files_only=False,
    )
    return model, tokenizer
