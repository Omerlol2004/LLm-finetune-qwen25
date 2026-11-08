"""FastAPI microservice for Qwen2.5-3B QLoRA + Confidence-Gated Hybrid RAG.

Provides four endpoints:
- ``GET  /health``     : liveness, KB status, and hardware/VRAM telemetry
- ``GET  /model-info`` : base model, LoRA adapter config, and KB metadata
- ``POST /retrieve``   : standalone confidence-gated retrieval from local KB
- ``POST /generate``   : end-to-end QLoRA + Confidence-Gated RAG chat completion
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engine import (  # noqa: E402
    DEFAULT_ADAPTER_DIR,
    DEFAULT_BASE_MODEL,
    DEFAULT_KB_DIR,
    DEFAULT_SYSTEM_PROMPT,
    bf16_supported,
    generate_response,
    load_qwen_model,
    vram_status,
)
from rag_utils import build_rag_prompt, init_kb  # noqa: E402

_tok: Any = None
_model: Any = None
_custom_generator: Optional[Callable[[str, str, int, float, bool], str]] = None


def set_custom_generator(fn: Optional[Callable[[str, str, int, float, bool], str]]) -> None:
    """Register an optional generator hook (used for deterministic CPU/unit testing)."""
    global _custom_generator
    _custom_generator = fn


def get_health() -> Dict[str, Any]:
    """Return service health and KB initialization telemetry."""
    kb_info = init_kb(DEFAULT_KB_DIR)
    return {
        "status": "ok",
        "model_loaded": bool(_model is not None or _custom_generator is not None),
        "bf16_supported": bf16_supported(),
        "vram": vram_status(),
        "kb": kb_info,
    }


def get_model_info() -> Dict[str, Any]:
    """Return base model ID, adapter configuration, and knowledge base metadata."""
    adapter_cfg_path = Path(DEFAULT_ADAPTER_DIR) / "adapter_config.json"
    if not adapter_cfg_path.exists():
        adapter_cfg_path = PROJECT_ROOT / "evidence" / "adapter_config.json"
    adapter_cfg: Dict[str, Any] = {}
    if adapter_cfg_path.exists():
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)

    kb_info = init_kb(DEFAULT_KB_DIR)
    return {
        "base_model": DEFAULT_BASE_MODEL,
        "adapter_dir": str(DEFAULT_ADAPTER_DIR),
        "adapter_config": {
            "peft_type": adapter_cfg.get("peft_type", "LORA"),
            "r": adapter_cfg.get("r", 8),
            "lora_alpha": adapter_cfg.get("lora_alpha", 16),
            "lora_dropout": adapter_cfg.get("lora_dropout", 0.05),
            "target_modules": adapter_cfg.get("target_modules", []),
        },
        "kb": kb_info,
    }


def handle_retrieve(
    query: str,
    top_k: int = 3,
    score_threshold: float = 0.30,
    strict_kb: bool = False,
) -> Dict[str, Any]:
    """Execute confidence-gated retrieval and prompt planning without LLM generation."""
    t0 = time.time()
    plan = build_rag_prompt(
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_query=query,
        use_rag=True,
        top_k=top_k,
        score_threshold=score_threshold,
        strict_kb=strict_kb,
        kb_dir=DEFAULT_KB_DIR,
    )
    return {
        "query": query,
        "mode": plan["mode"],
        "retrieved_summary": plan["retrieved_summary"],
        "hits": plan["hits"],
        "latency_ms": round((time.time() - t0) * 1000.0, 2),
    }


def handle_generate(
    query: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    use_rag: bool = True,
    top_k: int = 3,
    score_threshold: float = 0.30,
    strict_kb: bool = False,
    max_new_tokens: int = 160,
    temperature: float = 0.2,
    disable_adapter: bool = False,
) -> Dict[str, Any]:
    """Execute confidence-gated RAG + QLoRA generation."""
    global _tok, _model

    plan = build_rag_prompt(
        system_prompt=system_prompt,
        user_query=query,
        use_rag=use_rag,
        top_k=top_k,
        score_threshold=score_threshold,
        strict_kb=strict_kb,
        kb_dir=DEFAULT_KB_DIR,
    )

    t0 = time.time()
    if _custom_generator is not None:
        answer = _custom_generator(
            plan["system"],
            plan["user"],
            max_new_tokens,
            temperature,
            disable_adapter,
        )
    else:
        if _tok is None or _model is None:
            _tok, _model = load_qwen_model(
                DEFAULT_BASE_MODEL,
                DEFAULT_ADAPTER_DIR,
                load_adapter=True,
            )
        answer = generate_response(
            _tok,
            _model,
            system=plan["system"],
            user=plan["user"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            disable_adapter=disable_adapter,
        )

    return {
        "query": query,
        "answer": answer,
        "mode": plan["mode"],
        "retrieved_summary": plan["retrieved_summary"],
        "hits": plan["hits"],
        "latency_s": round(time.time() - t0, 3),
    }


try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    class RetrieveRequest(BaseModel):
        query: str = Field(..., min_length=1, description="User question to search against the KB")
        top_k: int = Field(default=3, ge=1, le=10)
        score_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
        strict_kb: bool = Field(default=False)

    class GenerateRequest(BaseModel):
        query: str = Field(..., min_length=1, description="User question or instruction")
        system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT)
        use_rag: bool = Field(default=True)
        top_k: int = Field(default=3, ge=1, le=10)
        score_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
        strict_kb: bool = Field(default=False)
        max_new_tokens: int = Field(default=160, ge=1, le=1024)
        temperature: float = Field(default=0.2, ge=0.0, le=2.0)
        disable_adapter: bool = Field(default=False)

    app = FastAPI(
        title="Qwen2.5-3B QLoRA + Confidence-Gated RAG API",
        version="2.0.0",
        description="4-bit NF4 QLoRA inference & confidence-gated FAISS RAG service.",
    )

    @app.get("/health")
    def health_endpoint() -> Dict[str, Any]:
        return get_health()

    @app.get("/model-info")
    def model_info_endpoint() -> Dict[str, Any]:
        return get_model_info()

    @app.post("/retrieve")
    def retrieve_endpoint(req: RetrieveRequest) -> Dict[str, Any]:
        return handle_retrieve(
            query=req.query,
            top_k=req.top_k,
            score_threshold=req.score_threshold,
            strict_kb=req.strict_kb,
        )

    @app.post("/generate")
    def generate_endpoint(req: GenerateRequest) -> Dict[str, Any]:
        return handle_generate(
            query=req.query,
            system_prompt=req.system_prompt,
            use_rag=req.use_rag,
            top_k=req.top_k,
            score_threshold=req.score_threshold,
            strict_kb=req.strict_kb,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            disable_adapter=req.disable_adapter,
        )

except ImportError:  # pragma: no cover
    app = None  # type: ignore
