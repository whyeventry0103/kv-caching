"""Remote-only lazy model loading."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


def _require_remote() -> None:
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise RuntimeError("model loading is remote-only; set KVCACHE_REMOTE=1")


def _revision(model_cfg: Mapping[str, Any]) -> str:
    revision = str(model_cfg.get("revision", ""))
    if not revision or revision == "TODO(human)":
        raise RuntimeError("model revision must be pinned before remote loading")
    return revision


def load_model_and_tokenizer(
    model_cfg: Mapping[str, Any],
    *,
    device: str = "cuda:0",
    allow_download: bool = False,
) -> tuple[Any, Any]:
    """Load one model on one GPU; network access is separately confirmed."""
    _require_remote()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for non-dry model execution")
    handle = str(model_cfg["handle"])
    revision = _revision(model_cfg)
    torch_dtype = getattr(torch, str(model_cfg.get("dtype", "bfloat16")))
    common = {
        "revision": revision,
        "local_files_only": not allow_download,
    }
    tokenizer = AutoTokenizer.from_pretrained(handle, **common)
    model = AutoModelForCausalLM.from_pretrained(
        handle,
        torch_dtype=torch_dtype,
        device_map={"": device},
        **common,
    )
    model.eval()
    return model, tokenizer


def load_tokenizer(
    model_cfg: Mapping[str, Any],
    *,
    allow_download: bool = False,
) -> Any:
    """Load only the tokenizer for a cheap remote chat-template probe."""
    _require_remote()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(model_cfg["handle"]),
        revision=_revision(model_cfg),
        local_files_only=not allow_download,
    )


def render_prompt(
    tokenizer: Any,
    question: str,
    model_cfg: Mapping[str, Any],
) -> str:
    """Render a single user turn with no system message."""
    if model_cfg.get("system_prompt") is not None:
        raise ValueError("pilot requires system_prompt: null")
    template_kwargs = dict(model_cfg.get("chat_template_kwargs", {}))
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )


def load_model(
    handle: str,
    *,
    dry_run: bool,
    remote: bool | None = None,
    dtype: str = "bfloat16",
) -> tuple[Any, Any] | None:
    """Compatibility wrapper retained for callers using the original scaffold."""
    if dry_run:
        return None
    if remote is False:
        raise RuntimeError("model loading is remote-only")
    return load_model_and_tokenizer(
        {"handle": handle, "dtype": dtype},
        allow_download=os.environ.get("KVCACHE_ALLOW_DOWNLOAD") == "1",
    )
