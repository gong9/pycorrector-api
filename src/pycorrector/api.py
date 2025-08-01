import os
import time
import logging
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import (
    CorrectionRequest,
    BatchCorrectionRequest,
    CorrectionResponse,
    BatchCorrectionResponse,
    CorrectionResult,
    ErrorInfo,
    ErrorResponse,
    HealthResponse,
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量存储模型实例
correctors: Dict[str, Any] = {}


def load_models():
    """加载纠错模型"""
    global correctors

    try:
        # 设置环境变量
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        # 禁用 MPS 检测
        import torch

        torch.backends.mps.is_available = lambda: False

        logger.info("开始加载纠错模型...")

        # 加载GPT纠错模型
        try:
            from pycorrector.gpt.gpt_corrector import GptCorrector

            correctors["gpt"] = GptCorrector(device="cpu")
            logger.info("GPT纠错模型加载成功")
        except Exception as e:
            logger.error(f"GPT纠错模型加载失败: {e}")
            correctors["gpt"] = None

        # 加载MacBert纠错模型
        try:
            from pycorrector.macbert.macbert_corrector import MacBertCorrector

            correctors["macbert"] = MacBertCorrector(
                "shibing624/macbert4csc-base-chinese"
            )
            logger.info("MacBert纠错模型加载成功")
        except Exception as e:
            logger.error(f"MacBert纠错模型加载失败: {e}")
            correctors["macbert"] = None

    except Exception as e:
        logger.error(f"模型加载过程中发生错误: {e}")


def unload_models():
    """卸载模型"""
    global correctors
    correctors.clear()
    logger.info("模型已卸载")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载模型
    load_models()
    yield
    # 关闭时卸载模型
    unload_models()


# 创建FastAPI应用
app = FastAPI(
    title="中文文本纠错API服务",
    description="基于深度学习的中文文本错误检测与纠正服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_errors(errors_list):
    """格式化错误信息"""
    formatted_errors = []
    for error in errors_list:
        if isinstance(error, tuple) and len(error) == 3:
            original, corrected, position = error
            formatted_errors.append(
                ErrorInfo(original=original, corrected=corrected, position=position)
            )
    return formatted_errors


def process_correction_result(result):
    """处理纠错结果"""
    if isinstance(result, dict):
        return CorrectionResult(
            source=result.get("source", ""),
            target=result.get("target", ""),
            errors=format_errors(result.get("errors", [])),
        )
    return result


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="内部服务器错误", detail=str(exc)).dict(),
    )


@app.get("/", response_model=dict)
async def root():
    """根路径"""
    return {"message": "中文文本纠错API服务", "version": "1.0.0", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    models_status = {}
    for model_name, model_instance in correctors.items():
        models_status[model_name] = model_instance is not None

    return HealthResponse(
        status="healthy" if any(models_status.values()) else "unhealthy",
        models_loaded=models_status,
        version="1.0.0",
    )


@app.post("/correct", response_model=CorrectionResponse)
async def correct_text(request: CorrectionRequest):
    """单个文本纠错"""
    start_time = time.time()

    try:
        # 检查模型是否可用
        if (
            request.model_type not in correctors
            or correctors[request.model_type] is None
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"模型 {request.model_type} 不可用",
            )

        corrector = correctors[request.model_type]

        # 执行纠错
        if hasattr(corrector, "correct"):
            # MacBert模型使用correct方法
            result = corrector.correct(request.text)
        else:
            # GPT模型使用correct_batch方法
            results = corrector.correct_batch([request.text])
            result = (
                results[0]
                if results
                else {"source": request.text, "target": request.text, "errors": []}
            )

        processing_time = time.time() - start_time

        return CorrectionResponse(
            success=True,
            data=process_correction_result(result),
            message="纠错完成",
            processing_time=round(processing_time, 3),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"纠错过程中发生错误: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"纠错失败: {str(e)}",
        )


@app.post("/correct/batch", response_model=BatchCorrectionResponse)
async def correct_batch_texts(request: BatchCorrectionRequest):
    """批量文本纠错"""
    start_time = time.time()

    try:
        # 检查模型是否可用
        if (
            request.model_type not in correctors
            or correctors[request.model_type] is None
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"模型 {request.model_type} 不可用",
            )

        corrector = correctors[request.model_type]
        results = []

        # 根据模型类型执行不同的纠错逻辑
        if hasattr(corrector, "correct_batch"):
            # GPT模型支持批量处理
            batch_results = corrector.correct_batch(request.texts)
            results = [process_correction_result(result) for result in batch_results]
        else:
            # MacBert模型逐个处理
            for text in request.texts:
                result = corrector.correct(text)
                results.append(process_correction_result(result))

        processing_time = time.time() - start_time

        return BatchCorrectionResponse(
            success=True,
            data=results,
            message="批量纠错完成",
            processing_time=round(processing_time, 3),
            total_count=len(results),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量纠错过程中发生错误: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量纠错失败: {str(e)}",
        )


@app.get("/models", response_model=dict)
async def list_models():
    """获取可用模型列表"""
    models_info = {}
    for model_name, model_instance in correctors.items():
        models_info[model_name] = {
            "loaded": model_instance is not None,
            "description": {
                "gpt": "基于GPT的中文文本纠错模型",
                "macbert": "基于MacBERT的中文拼写检查模型",
            }.get(model_name, "未知模型"),
        }

    return {"available_models": models_info, "default_model": "gpt"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
