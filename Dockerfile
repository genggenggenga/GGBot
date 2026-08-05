# GGBot 智能客服系统 — Docker 多阶段构建
# 目标：生产镜像尽量精简，开发镜像包含调试工具

# ── 阶段 1：基础环境 ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 切换为国内 apt + pip 镜像源，避免构建时网络超时
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends --fix-missing curl \
    || (apt-get update && apt-get install -y --no-install-recommends --fix-missing curl) \
    && rm -rf /var/lib/apt/lists/*

# ── 阶段 2：安装 Python 依赖 ──────────────────────────────────────────────────
FROM base AS dependencies

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ChromaDB 默认会用 ONNX all-MiniLM-L6-v2 做内部 embedding；但本项目走
# SiliconFlow API 路径（RAG_EMBEDDING_PROVIDER=api），不触发该模型加载，
# 因此不再预下载 79MB ONNX 权重。如改回本地 BGE，恢复下方注释段即可：
# RUN mkdir -p /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2 && \
#     curl -L --retry 3 --retry-delay 5 -o /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz \
#     https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz && \
#     cd /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2 && \
#     tar -xzf onnx.tar.gz && rm onnx.tar.gz

# ── 阶段 3：生产镜像 ──────────────────────────────────────────────────────────
FROM base AS production

# 非 root 用户运行。先创建用户，后续 COPY 直接带 owner，避免 chown -R 复制出额外大层。
RUN useradd -m -u 1000 ggbot

# 从依赖阶段复制已安装的包
COPY --from=dependencies /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# 复制应用代码
COPY --chown=ggbot:ggbot . .

# 创建必要目录，只调整运行期需要写入的目录权限，避免递归 chown 整个应用。
RUN mkdir -p /app/data/chroma /app/logs /app/config && \
    chown ggbot:ggbot /app/data /app/data/chroma /app/logs /app/config
USER ggbot

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── 阶段 4：开发镜像 ──────────────────────────────────────────────────────────
FROM dependencies AS development

COPY . .

RUN mkdir -p /app/data/chroma /app/logs /app/config /app/tests && \
    chmod -R 777 /app/data /app/logs

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
