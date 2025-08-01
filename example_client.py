#!/usr/bin/env python3
"""
中文文本纠错API服务客户端示例
使用示例：python example_client.py
"""

import requests
import json
import time


class CorrectorClient:
    """文本纠错API客户端"""

    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self):
        """健康检查"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"健康检查失败: {e}")
            return None

    def correct_text(self, text, model_type="gpt"):
        """单个文本纠错"""
        data = {"text": text, "model_type": model_type}

        try:
            response = self.session.post(
                f"{self.base_url}/correct",
                json=data,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"纠错请求失败: {e}")
            return None

    def correct_batch(self, texts, model_type="gpt"):
        """批量文本纠错"""
        data = {"texts": texts, "model_type": model_type}

        try:
            response = self.session.post(
                f"{self.base_url}/correct/batch",
                json=data,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"批量纠错请求失败: {e}")
            return None

    def list_models(self):
        """获取可用模型列表"""
        try:
            response = self.session.get(f"{self.base_url}/models")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"获取模型列表失败: {e}")
            return None


def main():
    """主函数"""
    client = CorrectorClient()

    print("=" * 50)
    print("中文文本纠错API服务客户端示例")
    print("=" * 50)

    # 1. 健康检查
    print("\n1. 健康检查:")
    health = client.health_check()
    if health:
        print(json.dumps(health, ensure_ascii=False, indent=2))
    else:
        print("服务不可用，请确保API服务正在运行")
        return

    # 2. 获取模型列表
    print("\n2. 可用模型:")
    models = client.list_models()
    if models:
        print(json.dumps(models, ensure_ascii=False, indent=2))

    # 3. 单个文本纠错示例
    print("\n3. 单个文本纠错:")
    test_text = "今天新情很好"

    for model_type in ["gpt", "macbert"]:
        print(f"\n使用 {model_type} 模型:")
        start_time = time.time()
        result = client.correct_text(test_text, model_type)
        if result:
            data = result.get("data", {})
            print(f"原文: {data.get('source', '')}")
            print(f"纠正: {data.get('target', '')}")
            if data.get("errors"):
                print(f"错误: {data.get('errors')}")
            print(f"处理时间: {result.get('processing_time', 0)}秒")
        else:
            print(f"{model_type} 模型不可用")

    # 4. 批量文本纠错示例
    print("\n4. 批量文本纠错:")
    test_texts = ["今天新情很好", "这就是生或啊", "我要去学校学习知识"]

    print(f"使用 gpt 模型批量处理 {len(test_texts)} 个文本:")
    start_time = time.time()
    batch_result = client.correct_batch(test_texts, "gpt")
    if batch_result:
        print(f"总处理时间: {batch_result.get('processing_time', 0)}秒")
        print(f"处理文本数量: {batch_result.get('total_count', 0)}")

        for i, data in enumerate(batch_result.get("data", []), 1):
            print(f"\n文本{i}:")
            print(f"  原文: {data.get('source', '')}")
            print(f"  纠正: {data.get('target', '')}")
            if data.get("errors"):
                print(f"  错误: {data.get('errors')}")


if __name__ == "__main__":
    main()
