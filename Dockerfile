# Dockerfile for BTC Quant Trading System
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

# 使用阿里云镜像源加速（可选，网络问题时启用）
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources || true

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY btc_quant/ ./btc_quant/
COPY config.yaml .
COPY run_backtest.py .
COPY run_live.py .

# 创建数据、模型、日志目录
RUN mkdir -p data models logs

# 默认命令（可被 docker-compose 覆盖）
CMD ["python", "run_live.py"]
