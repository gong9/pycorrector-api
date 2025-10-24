from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """运行时可配置参数，支持环境变量覆盖。"""

    gpt_device: str = "cpu"
    macbert_base_model: str = "shibing624/macbert4csc-base-chinese"
    confusion_path: str = str(Path(__file__).parent / "resources" / "confusions.txt")

    class Config:
        env_prefix = "PYCORRECTOR_"
