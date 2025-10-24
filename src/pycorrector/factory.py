import os
import logging
from typing import Dict


from .adapters import BaseCorrectorAdapter, GptAdapter, MacBertAdapter, KenLMAdapter
from .settings import Settings

logger = logging.getLogger(__name__)


def configure_environment() -> None:
    """在导入任何 PyTorch 之前配置环境，避免 MPS 相关问题。"""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    try:
        import torch  # noqa: F401

        # 禁用 MPS 检测，避免在不稳定环境下导致崩溃
        import torch as _torch

        _torch.backends.mps.is_available = lambda: False
    except Exception:
        # 环境中可能没有 torch，按需忽略
        pass


def build_correctors(settings: Settings) -> Dict[str, BaseCorrectorAdapter]:
    """构建并返回所有可用纠错器实例的字典。"""
    configure_environment()

    correctors: Dict[str, BaseCorrectorAdapter] = {}

    # GPT (深度学习 + 混淆词表后处理)
    try:
        correctors["gpt"] = GptAdapter(
            device=settings.gpt_device, confusion_path=settings.confusion_path
        )
        logger.info(f"✓ GPT 模型加载成功 (混淆词表: {settings.confusion_path})")
    except Exception as e:
        logger.error(f"✗ GPT 模型加载失败: {e}")
        correctors["gpt"] = None  # type: ignore[assignment]

    # MacBERT (深度学习 + 混淆词表后处理)
    try:
        correctors["macbert"] = MacBertAdapter(
            settings.macbert_base_model, confusion_path=settings.confusion_path
        )
        logger.info(f"✓ MacBERT 模型加载成功 (混淆词表: {settings.confusion_path})")
    except Exception as e:
        logger.error(f"✗ MacBERT 模型加载失败: {e}")
        correctors["macbert"] = None  # type: ignore[assignment]

    # KenLM (原生支持自定义混淆词表)
    try:
        correctors["kenlm"] = KenLMAdapter(confusion_path=settings.confusion_path)
        logger.info(f"✓ KenLM 模型加载成功 (混淆词表: {settings.confusion_path})")
    except Exception as e:
        logger.error(f"✗ KenLM 模型加载失败: {e}")
        correctors["kenlm"] = None  # type: ignore[assignment]

    return correctors
