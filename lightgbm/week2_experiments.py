#!/usr/bin/env python3
"""
第二周优化实验：标签与损失函数
测试不同配置组合对回测性能的影响
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

def run_experiment(
    min_rr_ratio: float,
    loss_function: str,
    experiment_name: str
):
    """运行单个实验"""
    logger.info("="*80)
    logger.info(f"实验: {experiment_name}")
    logger.info(f"  盈亏比阈值: {min_rr_ratio}")
    logger.info(f"  损失函数: {loss_function}")
    logger.info("="*80)
    
    # 加载数据
    cfg = load_config(Path('config.yaml'))
    klines = load_klines(cfg)
    
    # 划分数据集
    train_end = pd.Timestamp(cfg.raw['history_data']['train_end'])
    backtest_start = pd.Timestamp(cfg.raw['history_data']['backtest_start'])
    
    klines_train = klines[klines['open_time'] < train_end].reset_index(drop=True)
    klines_backtest = klines[klines['open_time'] >= backtest_start].reset_index(drop=True)
    
    # 构建特征
    feature_label_data_train = build_features_and_labels(cfg, klines_train)
    X_train = feature_label_data_train.features.reset_index(drop=True)
    
    feature_label_data_backtest = build_features_and_labels(cfg, klines_backtest)
    X_backtest = feature_label_data_backtest.features.reset_index(drop=True)
    
    # 构建标签（使用实验参数）
    label_builder = RiskRewardLabelBuilder(
        target_return=0.01,
        max_holding_period=50,
        min_rr_ratio=min_rr_ratio,  # 实验参数
        volatility_factor=1.0
    )
    
    labels_train = label_builder.build_labels(klines_train).reset_index(drop=True)
    labels_backtest = label_builder.build_labels(klines_backtest).reset_index(drop=True)
    
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
    
    min_len_backtest = min(len(X_backtest), len(labels_backtest), len(klines_backtest))
    X_backtest = X_backtest.iloc[:min_len_backtest]
    labels_backtest = labels_backtest.iloc[:min_len_backtest]
    klines_backtest = klines_backtest.iloc[:min_len_backtest]
    
    # 训练模型（使用实验的损失函数）
    strategy = TwoStageRiskRewardStrategy(loss_function=loss_function)
    strategy.train(
        X_train_split, labels_train_split,
        X_val, labels_val,
        rr_threshold=min_rr_ratio  # 使用实验的盈亏比阈值
    )
    
    # 预测（使用实验的盈亏比阈值）
    predictions = strategy.predict(
        X_backtest,
        rr_threshold=min_rr_ratio,
        prob_threshold=0.65
    )
    
    # 回测（使用正确的参数格式）
    results = run_backtest_with_triple_exit(
        cfg,
        klines_backtest,
        predictions['direction'],
        features=X_backtest,  # 传入特征以获取ATR
        position_ratios=None
    )
    
    # 提取关键指标（从回测结果中计算）
    trades = results.get('trades', [])
    
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
    
    stats = {
        'experiment': experiment_name,
        'min_rr_ratio': min_rr_ratio,
        'loss_function': loss_function,
        'total_return': results['total_return'],
        'max_drawdown': results.get('max_drawdown', 0.0),
        'win_rate': results['win_rate'],
        'total_trades': results['total_trades'],
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'profit_loss_ratio': profit_loss_ratio,
    }
    
    logger.info(f"\n结果:")
    logger.info(f"  总收益率: {stats['total_return']:.2f}%")
    logger.info(f"  最大回撤: {stats['max_drawdown']:.2f}%")
    logger.info(f"  胜率: {stats['win_rate']:.2f}%")
    logger.info(f"  交易次数: {stats['total_trades']}")
    logger.info(f"  平均盈利: {stats['avg_profit']:.2f}")
    logger.info(f"  平均亏损: {stats['avg_loss']:.2f}")
    logger.info(f"  盈亏比: {stats['profit_loss_ratio']:.2f}")
    logger.info("")
    
    return stats


def main():
    """运行所有实验"""
    logger.info("\n")
    logger.info("="*80)
    logger.info("第二周优化实验：标签与损失函数")
    logger.info("="*80)
    logger.info("\n")
    
    # 定义实验配置
    experiments = [
        # 基线（当前配置）
        {"min_rr_ratio": 1.5, "loss_function": "rmse", "name": "Baseline"},
        
        # 实验组1：调整盈亏比阈值
        {"min_rr_ratio": 2.0, "loss_function": "rmse", "name": "RR=2.0+RMSE"},
        {"min_rr_ratio": 2.5, "loss_function": "rmse", "name": "RR=2.5+RMSE"},
        
        # 实验组2：调整损失函数
        {"min_rr_ratio": 1.5, "loss_function": "fair", "name": "RR=1.5+Fair"},
        {"min_rr_ratio": 1.5, "loss_function": "huber", "name": "RR=1.5+Huber"},
        
        # 实验组3：组合优化
        {"min_rr_ratio": 2.0, "loss_function": "fair", "name": "RR=2.0+Fair"},
        {"min_rr_ratio": 2.0, "loss_function": "huber", "name": "RR=2.0+Huber"},
    ]
    
    all_results = []
    
    for i, exp in enumerate(experiments, 1):
        logger.info(f"\n[{i}/{len(experiments)}] 开始实验...")
        try:
            result = run_experiment(
                min_rr_ratio=exp["min_rr_ratio"],
                loss_function=exp["loss_function"],
                experiment_name=exp["name"]
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"实验失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总结果
    logger.info("\n")
    logger.info("="*80)
    logger.info("实验结果汇总")
    logger.info("="*80)
    
    results_df = pd.DataFrame(all_results)
    
    # 按总收益率排序
    results_df = results_df.sort_values('total_return', ascending=False)
    
    logger.info(f"\n{results_df.to_string(index=False)}")
    
    # 保存结果
    results_df.to_csv('backtest/week2_optimization_results.csv', index=False)
    logger.info(f"\n结果已保存到: backtest/week2_optimization_results.csv")
    
    # 找出最佳配置
    best = results_df.iloc[0]
    logger.info(f"\n🏆 最佳配置:")
    logger.info(f"  实验名称: {best['experiment']}")
    logger.info(f"  盈亏比阈值: {best['min_rr_ratio']}")
    logger.info(f"  损失函数: {best['loss_function']}")
    logger.info(f"  总收益率: {best['total_return']:.2f}%")
    logger.info(f"  胜率: {best['win_rate']:.2f}%")
    logger.info(f"  盈亏比: {best['profit_loss_ratio']:.2f}")
    
    logger.info("\n" + "="*80)


if __name__ == '__main__':
    main()
