#!/usr/bin/env python3
"""基线 LightGBM 多分类模型训练脚本

- 使用 btc_quant.features.build_features_and_labels 构建特征和标签
- 使用 btc_quant.model.train_model 训练多分类 LightGBM
- 使用 btc_quant.model.save_model 保存为 models/model_latest.pkl
"""

import sys
from pathlib import Path
import logging

# 确保可以导入本目录下的 btc_quant 包
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from btc_quant.config import load_config  # type: ignore
from btc_quant.data import load_klines, download_historical_klines  # type: ignore
from btc_quant.features import build_features_and_labels  # type: ignore
from btc_quant.model import train_model, save_model  # type: ignore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=" * 60)
    logger.info("基线 LightGBM 多分类模型 - 训练开始")
    logger.info("=" * 60)

    # 1. 加载配置
    cfg_path = PROJECT_ROOT / "config.yaml"
    cfg = load_config(cfg_path)
    logger.info("配置文件: %s", cfg_path)

    # 2. 加载或下载历史 K 线数据
    logger.info("\n[步骤1] 加载历史 K 线数据")
    try:
        klines = load_klines(cfg)
    except FileNotFoundError:
        logger.warning("本地未找到K线数据，开始下载...")
        download_historical_klines(cfg)
        klines = load_klines(cfg)

    logger.info("K线数量: %s", f"{len(klines):,}")
    logger.info("时间范围: %s ~ %s", klines.index[0], klines.index[-1])

    # 3. 构建特征和标签
    logger.info("\n[步骤2] 构建特征和标签")
    fl_data = build_features_and_labels(cfg, klines)
    logger.info("特征维度: %s", fl_data.features.shape)
    logger.info("标签分布:\n%s", fl_data.labels.value_counts().sort_index())

    # 4. 训练 LightGBM 多分类模型
    logger.info("\n[步骤3] 训练 LightGBM 多分类模型")
    trained = train_model(cfg, fl_data)

    # 5. 保存模型
    logger.info("\n[步骤4] 保存模型到 models/model_latest.pkl")
    out_path = save_model(cfg, trained, name="model_latest.pkl")
    logger.info("模型已保存: %s", out_path)

    logger.info("\n✅ 基线 LightGBM 模型训练完成")


if __name__ == "__main__":
    main()
