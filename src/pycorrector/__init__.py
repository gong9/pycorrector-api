import os
from pycorrector import Corrector

# 在导入任何 PyTorch 相关库之前设置环境变量
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

current_dir = os.path.dirname(__file__)
custom_txt_path = os.path.join(current_dir, "custom.txt")
test_txt_path = os.path.join(current_dir, "test.txt")


def load_test_sentences(file_path):
    sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
    return sentences


error_sentences = load_test_sentences(test_txt_path)

# for i, sentence in enumerate(error_sentences, 1):
#     print(f"{i}. {sentence}")


# # 测试默认配置
# m = Corrector()
# results = m.correct_batch(error_sentences)

# for result in results:
#     print(f"原文: {result['source']}")
#     print(f"纠正: {result['target']}")
#     if result["errors"]:
#         print(f"错误: {result['errors']}")
#     print("-" * 30)

# print("\n" + "=" * 50)
# print("使用自定义混淆词表进行纠错:")
# print("=" * 50)


# 可选：完全禁用 MPS 检测
import torch

torch.backends.mps.is_available = lambda: False

from pycorrector.gpt.gpt_corrector import GptCorrector

# 强制使用 CPU 设备
m = GptCorrector(device="cpu")
print(m.correct_batch(["这就是生或啊"]))
