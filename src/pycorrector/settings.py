from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """运行时可配置参数，支持环境变量覆盖。"""

    gpt_device: str = "cpu"
    macbert_base_model: str = "shibing624/macbert4csc-base-chinese"
    confusion_path: str = str(Path(__file__).parent / "resources" / "confusions.txt")

    # Qwen 大模型配置
    qwen_api_key: str = ""  # 从 PYCORRECTOR_QWEN_API_KEY 读取
    qwen_model: str = "qwen-turbo"  # 默认使用 qwen-turbo（性价比最高）
    qwen_max_workers: int = 5  # 千问 API 最大并发数
    enable_qwen: bool = True  # 是否启用千问模型

    class Config:
        env_prefix = "PYCORRECTOR_"
        env_file = ".env"  # 读取 .env 文件
        env_file_encoding = "utf-8"
        extra = "ignore"  # 忽略额外的环境变量
        case_sensitive = False  # 不区分大小写
