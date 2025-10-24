from typing import List, Dict, Any, Optional
from .utils import load_confusion_dict, apply_confusion_dict


class BaseCorrectorAdapter:
    """统一纠错适配接口，屏蔽不同底层模型差异。"""

    def __init__(self, confusion_path: Optional[str] = None):
        self.confusion_dict = {}
        if confusion_path:
            self.confusion_dict = load_confusion_dict(confusion_path)

    def _apply_confusion_post_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """应用混淆词表后处理。"""
        if not self.confusion_dict:
            return result

        target = result.get("target", result.get("source", ""))
        corrected_target, new_errors = apply_confusion_dict(target, self.confusion_dict)

        # 合并错误列表
        errors = result.get("errors", [])
        for wrong, correct, pos in new_errors:
            errors.append({"original": wrong, "corrected": correct, "position": pos})

        return {
            "source": result.get("source", ""),
            "target": corrected_target,
            "errors": errors,
        }

    def correct_text(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError

    def correct_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        raise NotImplementedError


class GptAdapter(BaseCorrectorAdapter):
    """适配 GPT 纠错器。"""

    def __init__(
        self, device: str = "cpu", confusion_path: Optional[str] = None
    ) -> None:
        super().__init__(confusion_path)
        from pycorrector.gpt.gpt_corrector import GptCorrector

        self.corrector = GptCorrector(device=device)

    def correct_text(self, text: str) -> Dict[str, Any]:
        batch_result = self.corrector.correct_batch([text])
        if not batch_result:
            result = {"source": text, "target": text, "errors": []}
        else:
            result = batch_result[0]
        return self._apply_confusion_post_process(result)

    def correct_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        results = self.corrector.correct_batch(texts)
        return [self._apply_confusion_post_process(r) for r in results]


class MacBertAdapter(BaseCorrectorAdapter):
    """适配 MacBERT 纠错器。"""

    def __init__(
        self, model_name_or_path: str, confusion_path: Optional[str] = None
    ) -> None:
        super().__init__(confusion_path)
        from pycorrector.macbert.macbert_corrector import MacBertCorrector

        self.corrector = MacBertCorrector(model_name_or_path)

    def correct_text(self, text: str) -> Dict[str, Any]:
        result = self.corrector.correct(text)
        return self._apply_confusion_post_process(result)

    def correct_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for text in texts:
            results.append(self.correct_text(text))
        return results


class KenLMAdapter(BaseCorrectorAdapter):
    """适配基于 KenLM 的 Corrector，原生支持自定义混淆词表。"""

    def __init__(self, confusion_path: Optional[str] = None) -> None:
        super().__init__(None)  # KenLM 自己处理混淆词表，不用后处理
        from pycorrector import Corrector

        self.corrector = Corrector()
        if confusion_path:
            self.corrector.set_custom_confusion_path_or_dict(confusion_path)

    def correct_text(self, text: str) -> Dict[str, Any]:
        # Corrector.correct() 返回字典格式
        result = self.corrector.correct(text)
        # 如果返回的是 (corrected, errors) 元组格式（旧版本）
        if isinstance(result, tuple):
            corrected, errors = result
            return {"source": text, "target": corrected, "errors": errors}
        # 如果返回的是字典格式（新版本）
        return result

    def correct_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        return self.corrector.correct_batch(texts)
