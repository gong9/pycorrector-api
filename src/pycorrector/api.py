import os
import time
import logging
from typing import Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .models import (
    CorrectionRequest,
    BatchCorrectionRequest,
    FullTextCorrectionRequest,
    CorrectionResponse,
    BatchCorrectionResponse,
    CorrectionResult,
    ErrorInfo,
    ErrorResponse,
    HealthResponse,
)
from .utils import format_errors
from .factory import build_correctors
from .settings import Settings
from .constants import DEFAULT_MODEL_DESCRIPTIONS

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = Settings()
correctors: Dict[str, Any] = {}


def load_models():
    """加载纠错模型（使用工厂 + 适配器）。"""
    global correctors
    logger.info("开始加载纠错模型...")
    correctors = build_correctors(settings)  # type: ignore[assignment]


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

# 挂载静态文件
static_dir = Path(__file__).parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"静态文件目录已挂载: {static_dir}")


def process_correction_result(result):
    """处理纠错结果"""
    if isinstance(result, dict):
        return CorrectionResult(
            source=result.get("source", ""),
            target=result.get("target", ""),
            errors=[
                ErrorInfo(
                    original=e["original"],
                    corrected=e["corrected"],
                    position=e["position"],
                    error_type=e.get("error_type", "typo"),
                    explanation=e.get("explanation", ""),
                )
                for e in format_errors(result.get("errors", []))
            ],
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


@app.get("/")
async def root():
    """根路径 - 重定向到测试页面"""
    return RedirectResponse(url="/static/index.html")


@app.get("/api")
async def api_info():
    """API信息"""
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
        result = corrector.correct_text(request.text)

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
        batch_results = corrector.correct_texts(request.texts)
        results = [process_correction_result(r) for r in batch_results]

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


def merge_correction_results(results_list):
    """合并多个模型的纠错结果，优先保留有详细说明的错误"""
    if not results_list:
        return []

    # 获取第一个模型的结果作为基础
    merged_results = []
    num_texts = len(results_list[0])

    for i in range(num_texts):
        # 收集所有模型对同一文本的结果
        text_results = [results[i] for results in results_list if i < len(results)]

        if not text_results:
            continue

        # 使用第一个结果的 source 和 target
        base_result = text_results[0]
        source = base_result.source
        target = base_result.target

        # 合并所有模型的错误信息（使用字典来智能合并）
        error_dict = {}  # key: (position, original, corrected), value: ErrorInfo

        for result in text_results:
            for error in result.errors:
                # 创建错误的唯一标识（位置+原文+纠正）
                error_key = (error.position, error.original, error.corrected)

                if error_key not in error_dict:
                    # 新错误，直接添加
                    error_dict[error_key] = error
                else:
                    # 已存在的错误，优先保留有 explanation 的版本
                    existing_error = error_dict[error_key]
                    # 如果新错误有更详细的说明，则替换
                    if error.explanation and len(error.explanation.strip()) > len(
                        existing_error.explanation.strip()
                    ):
                        error_dict[error_key] = error

        # 转换为列表并按位置排序
        all_errors = sorted(error_dict.values(), key=lambda e: e.position)

        # 如果有多个模型给出了纠正建议，更新 target
        if len(text_results) > 1 and all_errors:
            # 使用最后一个模型的 target（通常是最强的模型）
            target = text_results[-1].target

        merged_results.append(
            CorrectionResult(
                source=source,
                target=target,
                errors=all_errors,
            )
        )

    return merged_results


@app.post("/correct/fulltext", response_model=BatchCorrectionResponse)
async def correct_fulltext(request: FullTextCorrectionRequest):
    """全文纠错 - 自动按换行符切割文本并批量处理"""
    start_time = time.time()

    try:
        # 按换行符切割文本
        lines = request.text.split("\n")
        # 过滤掉空行
        non_empty_lines = [line for line in lines if line.strip()]

        if not non_empty_lines:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="文本不能全部为空",
            )

        if request.use_ensemble:
            # 使用模型融合：MacBERT(已包含规则兜底) + 千问
            # 注意：MacBertAdapter 已经通过 _apply_confusion_post_process 应用了规则（混淆词表）
            ensemble_models = ["macbert", "qwen"]
            all_results = []

            for model_name in ensemble_models:
                if model_name not in correctors or correctors[model_name] is None:
                    logger.warning(f"模型 {model_name} 不可用，跳过")
                    continue

                try:
                    corrector = correctors[model_name]
                    batch_results = corrector.correct_texts(non_empty_lines)
                    results = [process_correction_result(r) for r in batch_results]
                    all_results.append(results)
                    logger.info(
                        f"模型 {model_name} 处理完成，发现 {sum(len(r.errors) for r in results)} 个错误"
                    )
                except Exception as e:
                    logger.error(f"模型 {model_name} 处理失败: {e}")
                    continue

            if not all_results:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="所有融合模型均不可用",
                )

            # 合并结果
            results = merge_correction_results(all_results)
            message = (
                f"全文纠错完成（MacBERT+规则 + 千问，共 {len(all_results)} 个模型）"
            )
        else:
            # 使用单个模型
            if (
                request.model_type not in correctors
                or correctors[request.model_type] is None
            ):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"模型 {request.model_type} 不可用",
                )

            corrector = correctors[request.model_type]
            batch_results = corrector.correct_texts(non_empty_lines)
            results = [process_correction_result(r) for r in batch_results]
            message = "全文纠错完成"

        processing_time = time.time() - start_time

        return BatchCorrectionResponse(
            success=True,
            data=results,
            message=message,
            processing_time=round(processing_time, 3),
            total_count=len(results),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"全文纠错过程中发生错误: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"全文纠错失败: {str(e)}",
        )


@app.get("/models", response_model=dict)
async def list_models():
    """获取可用模型列表"""
    models_info = {}
    for model_name, model_instance in correctors.items():
        models_info[model_name] = {
            "loaded": model_instance is not None,
            "description": DEFAULT_MODEL_DESCRIPTIONS.get(model_name, "未知模型"),
        }

    return {"available_models": models_info, "default_model": "gpt"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
