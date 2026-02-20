# Dockerfile for BTC Quant Trading System
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1     PYTHONDONTWRITEBYTECODE=1     TZ=UTC

RUN apt-get update &&     apt-get install -y --no-install-recommends     gcc     g++     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制核心模块
COPY btc_quant/ ./btc_quant/

# 复制配置文件
COPY config.yaml .

# 复制运行脚本
COPY run_live_dynamic_exposure.py .

# 创建目录
RUN mkdir -p data models logs

CMD ["python", "run_live_dynamic_exposure.py"]
