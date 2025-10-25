from typing import List, Dict, Any, Optional
from .utils import load_confusion_dict, apply_confusion_dict


class BaseCorrectorAdapter:
    """统一纠错适配接口，屏蔽不同底层模型差异。"""

    def __init__(self, confusion_path: Optional[str] = None):
        self.confusion_dict = {}
        if confusion_path:
            self.confusion_dict = load_confusion_dict(confusion_path)

    def _apply_confusion_post_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """应用混淆词表后处理，并统一添加错误类型。"""
        if not self.confusion_dict:
            # 即使没有混淆词表，也要统一添加 error_type
            return self._add_error_type(result)

        target = result.get("target", result.get("source", ""))
        corrected_target, new_errors = apply_confusion_dict(target, self.confusion_dict)

        # 合并错误列表
        errors = result.get("errors", [])
        for wrong, correct, pos in new_errors:
            errors.append(
                {
                    "original": wrong,
                    "corrected": correct,
                    "position": pos,
                    "error_type": "typo",  # 混淆词表纠错都是错别字
                    "explanation": "",
                }
            )

        result = {
            "source": result.get("source", ""),
            "target": corrected_target,
            "errors": errors,
        }

        return self._add_error_type(result)

    def _add_error_type(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """给所有错误统一添加 error_type 字段（传统模型都是错别字）。"""
        errors = result.get("errors", [])
        normalized_errors = []

        for error in errors:
            # 如果是 tuple 格式 (original, corrected, position)，转换为 dict
            if isinstance(error, tuple):
                if len(error) >= 3:
                    original = error[0]
                    corrected = error[1]
                    # 生成简单的说明
                    explanation = self._generate_explanation(original, corrected)
                    normalized_errors.append(
                        {
                            "original": original,
                            "corrected": corrected,
                            "position": error[2],
                            "error_type": "typo",
                            "explanation": explanation,
                        }
                    )
            # 如果是 dict 格式，补充缺失的字段
            elif isinstance(error, dict):
                if "error_type" not in error:
                    error["error_type"] = "typo"
                if "explanation" not in error or not error["explanation"]:
                    # 如果没有说明，生成一个
                    original = error.get("original", "")
                    corrected = error.get("corrected", "")
                    error["explanation"] = self._generate_explanation(
                        original, corrected
                    )
                normalized_errors.append(error)

        result["errors"] = normalized_errors
        return result

    def _generate_explanation(self, original: str, corrected: str) -> str:
        """为错误生成简单的说明"""
        if not original or not corrected:
            return "字符错误"

        # 判断是否是常见的错别字
        if len(original) == 1 and len(corrected) == 1:
            return f"'{original}' 应为 '{corrected}'"

        return "字词错误"

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
            result = {"source": text, "target": corrected, "errors": errors}
        # 统一添加错误类型
        return self._add_error_type(result)

    def correct_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        results = self.corrector.correct_batch(texts)
        # 统一添加错误类型
        return [self._add_error_type(r) for r in results]
