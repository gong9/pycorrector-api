from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CorrectionRequest(BaseModel):
    """文本纠错请求模型"""

    text: str = Field(..., description="需要纠错的文本", min_length=1, max_length=10000)
    model_type: str = Field(
        "gpt", description="使用的纠错模型类型", pattern="^(gpt|macbert)$"
    )

    class Config:
        schema_extra = {"example": {"text": "今天新情很好", "model_type": "gpt"}}


class BatchCorrectionRequest(BaseModel):
    """批量文本纠错请求模型"""

    texts: List[str] = Field(
        ..., description="需要纠错的文本列表", min_items=1, max_items=100
    )
    model_type: str = Field(
        "gpt", description="使用的纠错模型类型", pattern="^(gpt|macbert)$"
    )

    class Config:
        schema_extra = {
            "example": {"texts": ["今天新情很好", "这就是生或啊"], "model_type": "gpt"}
        }


class ErrorInfo(BaseModel):
    """错误信息模型"""

    original: str = Field(..., description="原始错误字符")
    corrected: str = Field(..., description="纠正后的字符")
    position: int = Field(..., description="错误位置")


class CorrectionResult(BaseModel):
    """单个文本纠错结果模型"""

    source: str = Field(..., description="原始文本")
    target: str = Field(..., description="纠错后的文本")
    errors: List[ErrorInfo] = Field(default=[], description="检测到的错误列表")

    class Config:
        schema_extra = {
            "example": {
                "source": "今天新情很好",
                "target": "今天心情很好",
                "errors": [{"original": "新", "corrected": "心", "position": 2}],
            }
        }


class CorrectionResponse(BaseModel):
    """文本纠错响应模型"""

    success: bool = Field(True, description="请求是否成功")
    data: CorrectionResult = Field(..., description="纠错结果")
    message: str = Field("操作成功", description="响应消息")
    processing_time: float = Field(..., description="处理时间（秒）")


class BatchCorrectionResponse(BaseModel):
    """批量文本纠错响应模型"""

    success: bool = Field(True, description="请求是否成功")
    data: List[CorrectionResult] = Field(..., description="批量纠错结果")
    message: str = Field("操作成功", description="响应消息")
    processing_time: float = Field(..., description="处理时间（秒）")
    total_count: int = Field(..., description="处理的文本总数")


class ErrorResponse(BaseModel):
    """错误响应模型"""

    success: bool = Field(False, description="请求是否成功")
    error: str = Field(..., description="错误信息")
    detail: Optional[str] = Field(None, description="错误详情")


class HealthResponse(BaseModel):
    """健康检查响应模型"""

    status: str = Field("healthy", description="服务状态")
    models_loaded: Dict[str, bool] = Field(..., description="模型加载状态")
    version: str = Field("1.0.0", description="服务版本")
