from typing import List, Dict, Any, Tuple
from pathlib import Path


def format_errors(errors_list) -> List[Dict[str, Any]]:
    """将底层 (original, corrected, position) 元组列表转换为统一结构。"""
    formatted = []
    for error in errors_list:
        if isinstance(error, tuple) and len(error) == 3:
            # 元组格式：(original, corrected, position)
            original, corrected, position = error
            formatted.append(
                {"original": original, "corrected": corrected, "position": position}
            )
        elif isinstance(error, dict):
            # 字典格式：已经是目标格式，直接添加
            formatted.append(error)
    return formatted


def load_confusion_dict(confusion_path: str) -> Dict[str, str]:
    """加载混淆词表，返回 {错误词: 正确词} 字典。"""
    confusion_dict = {}
    path = Path(confusion_path)

    if not path.exists():
        return confusion_dict

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                wrong, correct = parts[0], parts[1]
                confusion_dict[wrong] = correct

    return confusion_dict


def apply_confusion_dict(
    text: str, confusion_dict: Dict[str, str]
) -> Tuple[str, List[Tuple[str, str, int]]]:
    """应用混淆词表纠正文本，返回 (纠正后文本, 错误列表)。"""
    corrected_text = text
    errors = []

    for wrong, correct in confusion_dict.items():
        if wrong in corrected_text:
            # 找到所有出现位置
            start = 0
            while True:
                pos = corrected_text.find(wrong, start)
                if pos == -1:
                    break
                errors.append((wrong, correct, pos))
                corrected_text = corrected_text.replace(wrong, correct, 1)
                start = pos + len(correct)

    return corrected_text, errors
