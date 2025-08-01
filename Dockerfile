# 使用Python 3.9官方镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app/src
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装PDM
RUN pip install --no-cache-dir pdm

# 复制项目文件
COPY pyproject.toml pdm.lock* ./
COPY src/ src/
COPY README.md ./

# 安装Python依赖
RUN pdm install --no-dev

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["pdm", "run", "start"] 