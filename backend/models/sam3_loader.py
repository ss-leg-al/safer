"""SAM 3 (facebookresearch/sam3) singleton loader.

Loaded once at FastAPI startup so every job reuses the same GPU-resident model.
Per CLAUDE.md: re-loading per request would exhaust VRAM and add tens of seconds latency.

API used by tools (Meta's official SAM3 from facebookresearch/sam3):

    from .models.sam3_loader import get_sam3_processor
    processor = get_sam3_processor()
    state = processor.set_image(pil_image)
    output = processor.set_text_prompt(state=state, prompt="human face")
    masks  = output["masks"]   # torch.Tensor [N, 1, H, W] (binary)
    boxes  = output["boxes"]   # torch.Tensor [N, 4] (xyxy)
    scores = output["scores"]  # torch.Tensor [N]
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_model = None
_processor = None
_load_error: str | None = None
_checkpoint_path = Path("checkpoints/sam3.pt")


def is_available() -> bool:
    return _model is not None and _processor is not None


def get_load_error() -> str | None:
    return _load_error


def load_sam3(force: bool = False):
    """Load SAM3 onto GPU. Returns the processor (or None if checkpoint missing)."""
    global _model, _processor, _load_error
    if _processor is not None and not force:
        return _processor

    if not _checkpoint_path.exists():
        _load_error = (
            f"SAM3 checkpoint not found at {_checkpoint_path.resolve()}. "
            "Request access at https://huggingface.co/facebook/sam3, "
            f"download sam3.pt, then place it at {_checkpoint_path}."
        )
        logger.warning(_load_error)
        return None

    try:
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError as e:
        _load_error = (
            f"Meta SAM3 package not installed: {e}. "
            "Activate the 'sam3' conda env (Python 3.12) where sam3 is installed via "
            "pip install -e /home/hyunseo/sam3_repo."
        )
        logger.error(_load_error)
        return None

    if not torch.cuda.is_available():
        logger.warning("SAM3 will run on CPU — expect very slow inference.")

    # NOTE: SAM3 inference requires `with torch.autocast("cuda", dtype=torch.bfloat16):`
    # in the call site (see _sam3_detect in tools/agentic.py). autocast is thread/coroutine
    # local so we cannot enter it once here and have it apply to BackgroundTasks workers.

    try:
        model = build_sam3_image_model(
            checkpoint_path=str(_checkpoint_path),
            load_from_HF=True,  # for the BPE tokenizer (needs HF auth via env or hf auth login)
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        processor = Sam3Processor(model)
        _model = model
        _processor = processor
        _load_error = None
        logger.info("SAM3 (Meta) loaded on GPU with bfloat16 autocast")
        return _processor
    except Exception as e:
        _load_error = f"SAM3 load failed: {e}"
        logger.exception(_load_error)
        return None


def get_sam3_processor():
    if _processor is None:
        raise RuntimeError(_load_error or "SAM3 not loaded. Call load_sam3() first.")
    return _processor
