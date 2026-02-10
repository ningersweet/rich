#!/usr/bin/env python3
"""
测试单个实验配置，验证代码无误
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from btc_quant.config import load_config
from btc_quant.data import load_klines
from btc_quant.features import build_features_and_labels
from btc_quant.risk_reward_labels import RiskRewardLabelBuilder
from btc_quant.risk_reward_model import TwoStageRiskRewardStrategy
from btc_quant.backtest import run_backtest_with_triple_exit

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_baseline():
    """测试基线配置"""
    logger.info("="*80)
    logger.info("测试实验: Baseline (RR=1.5 + RMSE)")
    logger.info("="*80)
    
    # 加载数据
    cfg = load_config(Path('config.yaml'))
    klines = load_klines(cfg)
    
    # 划分数据集
    train_end = pd.Timestamp(cfg.raw['history_data']['train_end'])
    backtest_start = pd.Timestamp(cfg.raw['history_data']['backtest_start'])
    
    klines_train = klines[klines['open_time'] < train_end].reset_index(drop=True)
    klines_backtest = klines[klines['open_time'] >= backtest_start].reset_index(drop=True)
    
    logger.info(f"训练集K线数: {len(klines_train)}")
    logger.info(f"回测集K线数: {len(klines_backtest)}")
    
    # 构建特征
    feature_label_data_train = build_features_and_labels(cfg, klines_train)
    X_train = feature_label_data_train.features.reset_index(drop=True)
    
    feature_label_data_backtest = build_features_and_labels(cfg, klines_backtest)
    X_backtest = feature_label_data_backtest.features.reset_index(drop=True)
    
    logger.info(f"训练集特征数: {len(X_train)}, 特征维度: {X_train.shape[1]}")
    logger.info(f"回测集特征数: {len(X_backtest)}")
    
    # 构建标签
    label_builder = RiskRewardLabelBuilder(
        target_return=0.01,
        max_holding_period=50,
        min_rr_ratio=1.5,
        volatility_factor=1.0
    )
    
    labels_train = label_builder.build_labels(klines_train).reset_index(drop=True)
    labels_backtest = label_builder.build_labels(klines_backtest).reset_index(drop=True)
    
    logger.info(f"训练集标签数: {len(labels_train)}")
    logger.info(f"回测集标签数: {len(labels_backtest)}")
    
    # 对齐数据
    min_len_train = min(len(X_train), len(labels_train), len(klines_train))
    X_train = X_train.iloc[:min_len_train]
    labels_train = labels_train.iloc[:min_len_train]
    klines_train = klines_train.iloc[:min_len_train]
    
    # 划分训练集和验证集（80%训练，20%验证）
    split_idx = int(len(X_train) * 0.8)
    X_train_split = X_train.iloc[:split_idx].reset_index(drop=True)
    X_val = X_train.iloc[split_idx:].reset_index(drop=True)
    labels_train_split = labels_train.iloc[:split_idx].reset_index(drop=True)
    labels_val = labels_train.iloc[split_idx:].reset_index(drop=True)
    
    logger.info(f"训练集: {len(X_train_split)}, 验证集: {len(X_val)}")
    
    min_len_backtest = min(len(X_backtest), len(labels_backtest), len(klines_backtest))
    X_backtest = X_backtest.iloc[:min_len_backtest]
    labels_backtest = labels_backtest.iloc[:min_len_backtest]
    klines_backtest = klines_backtest.iloc[:min_len_backtest]
    
    logger.info(f"对齐后回测集: {len(klines_backtest)}")
    
    # 训练模型
    logger.info("\n开始训练模型...")
    strategy = TwoStageRiskRewardStrategy(loss_function='rmse')
    strategy.train(
        X_train_split, labels_train_split,
        X_val, labels_val,
        rr_threshold=1.5
    )
    
    # 预测
    logger.info("\n开始预测...")
    predictions = strategy.predict(
        X_backtest,
        rr_threshold=1.5,
        prob_threshold=0.65
    )
    
    logger.info(f"预测完成，信号数: {len(predictions)}")
    logger.info(f"做多信号: {(predictions['direction']==1).sum()}")
    logger.info(f"做空信号: {(predictions['direction']==-1).sum()}")
    logger.info(f"观望信号: {(predictions['direction']==0).sum()}")
    
    # 回测
    logger.info("\n开始回测...")
    results = run_backtest_with_triple_exit(
        cfg,
        klines_backtest,
        predictions['direction'],
        features=X_backtest,
        position_ratios=None
    )
    
    # 提取指标
    trades = results.get('trades', [])
    logger.info(f"\n回测完成，交易次数: {len(trades)}")
    
    # 计算平均盈利、平均亏损、盈亏比
    if trades:
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        avg_profit = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0.0
    else:
        avg_profit = 0.0
        avg_loss = 0.0
        profit_loss_ratio = 0.0
    
    # 输出结果
    logger.info("\n" + "="*80)
    logger.info("测试结果")
    logger.info("="*80)
    logger.info(f"总收益率: {results['total_return']:.2f}%")
    logger.info(f"最大回撤: {results.get('max_drawdown', 0.0):.2f}%")
    logger.info(f"胜率: {results['win_rate']:.2f}%")
    logger.info(f"交易次数: {results['total_trades']}")
    logger.info(f"平均盈利: {avg_profit:.2f}")
    logger.info(f"平均亏损: {avg_loss:.2f}")
    logger.info(f"盈亏比: {profit_loss_ratio:.2f}")
    logger.info("="*80)
    
    return results

if __name__ == '__main__':
    test_baseline()
