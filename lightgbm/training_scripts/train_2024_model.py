#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
LightGBM 两阶段风险收益模型训练脚本
===============================================================================

【功能说明】
训练基于2018-2024年历史数据的BTC量化交易模型，用于2025-2026年样本外回测验证。

【训练流程】
1. 加载并过滤2018-2024年K线数据
2. 构建67个技术指标特征
3. 构建风险收益标签（盈亏比、方向、持仓周期）
4. 两阶段训练：
   - 阶段1：使用全部67个特征训练，获取特征重要性排序
   - 阶段2：提取Top30特征重新训练，提高泛化能力
5. 保存模型到 models/final_2024_dynamic/

【数据划分】
- 训练数据：2018-01-01 至 2024-12-31（7年历史数据）
- 回测数据：2025-01-01 至 2026-02-20（样本外验证）
- 严格时间分离，避免数据泄露

【模型结构】
两阶段LightGBM模型：
- Stage1：预测盈亏比（risk_reward_model.txt）
- Stage2：预测方向和持仓周期（direction_model.txt, period_model.txt）

【输出文件】
models/final_2024_dynamic/
├── risk_reward_model.txt      # 盈亏比预测模型
├── direction_model.txt         # 方向预测模型
├── period_model.txt            # 周期预测模型
├── top30_features.txt          # Top30特征列表
└── training_info.txt           # 训练元信息

【使用方法】
cd /Users/lemonshwang/project/rich/lightgbm
python training_scripts/train_2024_model.py

【作者】Qoder AI
【日期】2026-02-21
===============================================================================
"""

import os
import sys

# 添加父目录到路径，以便导入btc_quant模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import TimeSeriesSplit

from btc_quant.config import load_config
from btc_quant.data import load_klines
from btc_quant.features import build_features_and_labels
from btc_quant.risk_reward_labels import RiskRewardLabelBuilder
from btc_quant.risk_reward_model import TwoStageRiskRewardStrategy

# ============================================================================
# 日志配置
# ============================================================================
from datetime import datetime as dt
log_file = Path('../logs') / f'train_2024_model_{dt.now().strftime("%Y%m%d_%H%M%S")}.log'
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"日志文件: {log_file.absolute()}")


# ============================================================================
# 常量配置
# ============================================================================
# 训练数据时间范围（硬编码，确保可复现）
TRAIN_START = '2018-01-01T00:00:00Z'
TRAIN_END = '2024-12-31T23:59:59Z'

# 模型保存目录
MODEL_DIR = Path('models/final_2024_dynamic')

# 标签构建参数
TARGET_RETURN = 0.01          # 目标收益率：1%
MAX_HOLDING_PERIOD = 50       # 最大持仓周期：50根K线（12.5小时，15分钟周期）
MIN_RR_RATIO = 1.5            # 最小盈亏比：1.5

# 训练参数
TRAIN_VAL_SPLIT = 0.8         # 训练/验证集划分比例
RR_THRESHOLD = 2.0            # 盈亏比阈值（训练时用）
TOP_N_FEATURES = 30           # Top特征数量


def main():
    """
    主训练流程
    
    Returns:
        None
    """
    logger.info("="*80)
    logger.info("LightGBM 两阶段风险收益模型训练")
    logger.info("="*80)
    
    # ------------------------------------------------------------------------
    # 步骤1：加载配置和数据
    # ------------------------------------------------------------------------
    logger.info(f"\n训练数据范围: {TRAIN_START} 至 {TRAIN_END}")
    
    cfg = load_config()
    logger.info("\n[1/6] 加载K线数据...")
    klines_all = load_klines(cfg)
    klines_all['close_time'] = pd.to_datetime(klines_all['close_time'])
    
    logger.info(f"原始数据: {len(klines_all)} 根K线")
    logger.info(f"时间跨度: {klines_all['close_time'].min()} 至 {klines_all['close_time'].max()}")
    
    # ------------------------------------------------------------------------
    # 步骤2：过滤训练集数据
    # ------------------------------------------------------------------------
    logger.info("\n[2/6] 过滤训练集数据...")
    train_start_ts = pd.Timestamp(TRAIN_START)
    train_end_ts = pd.Timestamp(TRAIN_END)
    
    klines_train = klines_all[
        (klines_all['close_time'] >= train_start_ts) & 
        (klines_all['close_time'] <= train_end_ts)
    ].reset_index(drop=True)
    
    logger.info(f"训练集K线: {len(klines_train)} 根")
    logger.info(f"训练集时间: {klines_train['close_time'].min()} 至 {klines_train['close_time'].max()}")
    
    # ------------------------------------------------------------------------
    # 步骤3：构建特征和标签
    # ------------------------------------------------------------------------
    logger.info("\n[3/6] 构建技术指标特征...")
    feature_label_data = build_features_and_labels(cfg, klines_train)
    X_train_all = feature_label_data.features.reset_index(drop=True)
    
    logger.info(f"特征数量: {X_train_all.shape[1]} 个")
    logger.info(f"特征样本: {len(X_train_all)} 个")
    
    logger.info("\n构建风险收益标签...")
    label_builder = RiskRewardLabelBuilder(
        target_return=TARGET_RETURN,
        max_holding_period=MAX_HOLDING_PERIOD,
        min_rr_ratio=MIN_RR_RATIO
    )
    
    labels_df = label_builder.build_labels(klines_train)
    logger.info(f"标签样本: {len(labels_df)} 个")
    
    # 对齐特征和标签（因为特征计算有窗口期，会损失部分样本）
    min_len = min(len(X_train_all), len(labels_df))
    X_train_all = X_train_all.iloc[:min_len].reset_index(drop=True)
    labels_train_all = labels_df.iloc[:min_len].reset_index(drop=True)
    
    logger.info(f"对齐后样本: {len(X_train_all)} 个")
    
    # ------------------------------------------------------------------------
    # 步骤4：划分训练集和验证集（使用时间序列交叉验证）
    # ------------------------------------------------------------------------
    logger.info("\n[4/6] 划分训练集和验证集...")
    
    # 使用时间序列交叉验证，取最后一个fold作为验证集
    tscv = TimeSeriesSplit(n_splits=5)
    train_indices = []
    val_indices = []
    
    for train_idx, val_idx in tscv.split(X_train_all):
        train_indices = train_idx
        val_indices = val_idx
    
    # 使用最后一个fold作为验证集
    X_train = X_train_all.iloc[train_indices]
    X_val = X_train_all.iloc[val_indices]
    labels_train = labels_train_all.iloc[train_indices]
    labels_val = labels_train_all.iloc[val_indices]
    
    train_pct = len(X_train) / len(X_train_all) * 100
    val_pct = len(X_val) / len(X_train_all) * 100
    
    logger.info(f"训练集: {len(X_train)} 个样本 ({train_pct:.1f}%)")
    logger.info(f"验证集: {len(X_val)} 个样本 ({val_pct:.1f}%)")
    logger.info(f"验证集时间范围: {klines_train.iloc[val_indices]['close_time'].min()} 至 {klines_train.iloc[val_indices]['close_time'].max()}")
    
    # ------------------------------------------------------------------------
    # 步骤5：两阶段训练
    # ------------------------------------------------------------------------
    logger.info("\n[5/6] 开始两阶段训练...")
    
    # ---- 阶段1：全特征训练，获取特征重要性 ----
    logger.info("\n" + "="*80)
    logger.info("阶段1：全特征训练（67个特征）")
    logger.info("="*80)
    
    strategy_full = TwoStageRiskRewardStrategy()
    train_results_full = strategy_full.train(
        X_train=X_train,
        labels_train=labels_train,
        X_val=X_val,
        labels_val=labels_val,
        rr_threshold=RR_THRESHOLD
    )
    
    # 提取特征重要性（直接从模型获取）
    logger.info("\n提取特征重要性...")
    
    # 从盈亏比模型获取特征重要性
    rr_importance = strategy_full.rr_model.model.feature_importance(importance_type='gain')
    feature_names = strategy_full.rr_model.model.feature_name()
    
    # 检查特征名称和重要性数组长度是否一致
    if len(feature_names) != len(rr_importance):
        logger.warning(f"特征名称数量({len(feature_names)})与重要性数组长度({len(rr_importance)})不一致！")
        # 取最小长度以确保匹配
        min_len = min(len(feature_names), len(rr_importance))
        feature_names = feature_names[:min_len]
        rr_importance = rr_importance[:min_len]
        logger.warning(f"已截断至{min_len}个特征")
    
    importance_dict = dict(zip(feature_names, rr_importance))
    sorted_features = sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # 提取Top30特征
    top_30_features = [f[0] for f in sorted_features[:TOP_N_FEATURES]]
    
    logger.info(f"\nTop{TOP_N_FEATURES}特征（按重要性排序）:")
    for i, (feat, imp) in enumerate(sorted_features[:TOP_N_FEATURES], 1):
        logger.info(f"  {i:2d}. {feat:35s} : {imp:8.0f}")
    
    # ---- 阶段2：Top30特征重新训练 ----
    logger.info("\n" + "="*80)
    logger.info(f"阶段2：Top{TOP_N_FEATURES}特征重新训练")
    logger.info("="*80)
    
    X_train_top30 = X_train[top_30_features]
    X_val_top30 = X_val[top_30_features]
    
    strategy_top30 = TwoStageRiskRewardStrategy()
    train_results_top30 = strategy_top30.train(
        X_train=X_train_top30,
        labels_train=labels_train,
        X_val=X_val_top30,
        labels_val=labels_val,
        rr_threshold=RR_THRESHOLD
    )
    
    logger.info("\n✅ 两阶段训练完成")
    
    # ------------------------------------------------------------------------
    # 步骤6：保存模型和元信息
    # ------------------------------------------------------------------------
    logger.info("\n[6/6] 保存模型...")
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 保存模型文件
    strategy_top30.save(MODEL_DIR)
    logger.info(f"✅ 模型已保存: {MODEL_DIR}")
    
    # 保存Top30特征列表
    with open(MODEL_DIR / 'top30_features.txt', 'w') as f:
        for feat in top_30_features:
            f.write(f"{feat}\n")
    logger.info(f"✅ 特征列表已保存: {MODEL_DIR / 'top30_features.txt'}")
    
    # 保存训练元信息
    with open(MODEL_DIR / 'training_info.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("训练配置信息\n")
        f.write("="*80 + "\n\n")
        
        f.write("[数据信息]\n")
        f.write(f"训练时间范围: {TRAIN_START} 至 {TRAIN_END}\n")
        f.write(f"训练样本数: {len(X_train):,}\n")
        f.write(f"验证样本数: {len(X_val):,}\n")
        f.write(f"总K线数: {len(klines_train):,}\n\n")
        
        f.write("[特征信息]\n")
        f.write(f"原始特征数: {X_train_all.shape[1]}\n")
        f.write(f"选择特征数: {len(top_30_features)}\n")
        f.write(f"特征选择方法: LightGBM特征重要性Top30\n\n")
        
        f.write("[标签参数]\n")
        f.write(f"目标收益率: {TARGET_RETURN*100}%\n")
        f.write(f"最大持仓周期: {MAX_HOLDING_PERIOD}根K线\n")
        f.write(f"最小盈亏比: {MIN_RR_RATIO}\n\n")
        
        f.write("[训练参数]\n")
        f.write(f"盈亏比阈值: {RR_THRESHOLD}\n")
        f.write(f"训练/验证比例: {TRAIN_VAL_SPLIT}/{1-TRAIN_VAL_SPLIT}\n\n")
        
        f.write("[回测建议参数]\n")
        f.write(f"- 盈亏比阈值: 2.5\n")
        f.write(f"- 方向概率阈值: 0.70\n")
        f.write(f"- 复利模式: 启用\n")
        f.write(f"- 动态敞口范围: 1-10倍\n")
        f.write(f"- 固定止损: -3%\n")
        f.write(f"- 追踪止损: 启用\n")
        f.write(f"- 每日最大亏损: -20%\n")
        f.write(f"- 回撤暂停阈值: 10%\n\n")
        
        f.write("[模型文件]\n")
        f.write(f"- risk_reward_model.txt  (盈亏比预测)\n")
        f.write(f"- direction_model.txt    (方向预测)\n")
        f.write(f"- period_model.txt       (周期预测)\n")
        f.write(f"- top30_features.txt     (特征列表)\n")
    
    logger.info(f"✅ 训练信息已保存: {MODEL_DIR / 'training_info.txt'}")
    
    # ------------------------------------------------------------------------
    # 完成
    # ------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("✅ 训练完成！")
    logger.info("="*80)
    logger.info(f"\n📁 模型保存位置: {MODEL_DIR.absolute()}")
    logger.info("\n🚀 下一步：运行回测脚本")
    logger.info(f"   cd /Users/lemonshwang/project/rich/lightgbm")
    logger.info(f"   python backtest_scripts/backtest_2024_model.py  # 或 python backtest_scripts/backtest_engine.py")
    logger.info("="*80)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
