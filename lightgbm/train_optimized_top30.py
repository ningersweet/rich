#!/usr/bin/env python3
"""
优化后的盈亏比驱动策略 - Top-30特征版本

配置：
- RR阈值：2.0
- 损失函数：RMSE
- 特征数量：Top-30（从67个中筛选）
- 回测逻辑：简化回测（10%仓位，按预测周期平仓）

预期效果：
- 总收益率：~123.65%
- 胜率：~79%
- 盈亏比：~3.10
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

from btc_quant.config import load_config
from btc_quant.data import load_klines
from btc_quant.features import build_features_and_labels
from btc_quant.risk_reward_labels import RiskRewardLabelBuilder
from btc_quant.risk_reward_model import TwoStageRiskRewardStrategy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('optimized_rr_strategy_top30.log')
    ]
)
logger = logging.getLogger(__name__)


def get_top_30_features(strategy):
    """获取Top-30重要特征"""
    rr_importance = strategy.rr_model.model.feature_importance(importance_type='gain')
    feature_names = strategy.rr_model.model.feature_name()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rr_importance
    }).sort_values('importance', ascending=False)
    
    top_30_features = importance_df.head(30)['feature'].tolist()
    
    logger.info(f"\nTop-30 重要特征:")
    for i, row in importance_df.head(30).iterrows():
        logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.0f}")
    
    return top_30_features


def simple_backtest(klines, predictions, initial_balance=1000.0, position_size_pct=0.10):
    """简化回测逻辑（第一周成功配置）"""
    equity = initial_balance
    trades = []
    position = None
    
    for i in range(len(predictions)):
        # 检查是否应该平仓
        if position is not None:
            bars_held = i - position['entry_idx']
            current_price = klines.iloc[i]['close']
            
            # 达到预期持有周期，平仓
            if bars_held >= position['hold_period']:
                if position['side'] == 1:
                    pnl = (current_price - position['entry_price']) * position['quantity']
                else:
                    pnl = (position['entry_price'] - current_price) * position['quantity']
                
                equity += pnl
                trades.append({
                    'entry_time': klines.iloc[position['entry_idx']]['open_time'],
                    'exit_time': klines.iloc[i]['open_time'],
                    'side': 'long' if position['side'] == 1 else 'short',
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'quantity': position['quantity'],
                    'pnl': pnl
                })
                position = None
        
        # 检查是否应该开仓
        if position is None and predictions.iloc[i]['should_trade']:
            entry_price = klines.iloc[i]['close']
            quantity = (equity * position_size_pct) / entry_price
            
            position = {
                'side': predictions.iloc[i]['direction'],
                'entry_price': entry_price,
                'entry_idx': i,
                'hold_period': int(predictions.iloc[i]['holding_period']),
                'quantity': quantity
            }
    
    # 最后平仓
    if position is not None:
        final_price = klines.iloc[-1]['close']
        if position['side'] == 1:
            pnl = (final_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - final_price) * position['quantity']
        
        equity += pnl
        trades.append({
            'entry_time': klines.iloc[position['entry_idx']]['open_time'],
            'exit_time': klines.iloc[-1]['open_time'],
            'side': 'long' if position['side'] == 1 else 'short',
            'entry_price': position['entry_price'],
            'exit_price': final_price,
            'quantity': position['quantity'],
            'pnl': pnl
        })
    
    # 计算统计指标
    total_return = (equity / initial_balance - 1) * 100
    trades_df = pd.DataFrame(trades)
    
    if len(trades) > 0:
        winning_trades = (trades_df['pnl'] > 0).sum()
        losing_trades = (trades_df['pnl'] <= 0).sum()
        win_rate = winning_trades / len(trades) * 100
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].mean()) if losing_trades > 0 else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    else:
        win_rate = 0
        winning_trades = 0
        losing_trades = 0
        avg_win = 0
        avg_loss = 0
        profit_loss_ratio = 0
    
    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'total_trades': len(trades),
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'profit_loss_ratio': profit_loss_ratio,
        'avg_profit': avg_win,
        'avg_loss': avg_loss,
        'trades': trades
    }


def main():
    """主流程"""
    logger.info("=" * 100)
    logger.info("优化后的盈亏比驱动策略 - Top-30特征版本")
    logger.info("=" * 100)
    
    # 加载配置
    cfg_path = Path(__file__).parent / 'config.yaml'
    cfg = load_config(cfg_path)
    
    # 加载K线数据
    logger.info("\n【步骤1】加载K线数据")
    klines = load_klines(cfg)
    logger.info(f"总K线数: {len(klines):,}")
    
    # 划分数据集
    logger.info("\n【步骤2】划分数据集")
    train_end = pd.Timestamp(cfg.raw['history_data']['train_end'])
    backtest_start = pd.Timestamp(cfg.raw['history_data']['backtest_start'])
    
    klines_train = klines[klines['open_time'] < train_end]
    klines_backtest = klines[klines['open_time'] >= backtest_start]
    
    logger.info(f"训练集: {len(klines_train):,} K线")
    logger.info(f"回测集: {len(klines_backtest):,} K线")
    
    # 构建特征
    logger.info("\n【步骤3】构建特征")
    feature_label_data_train = build_features_and_labels(cfg, klines_train)
    X_train_full = feature_label_data_train.features.reset_index(drop=True)
    
    feature_label_data_backtest = build_features_and_labels(cfg, klines_backtest)
    X_backtest_full = feature_label_data_backtest.features.reset_index(drop=True)
    
    logger.info(f"训练特征: {X_train_full.shape}")
    logger.info(f"回测特征: {X_backtest_full.shape}")
    
    # 构建盈亏比标签
    logger.info("\n【步骤4】构建盈亏比标签")
    label_builder = RiskRewardLabelBuilder(
        target_return=0.01,
        max_holding_period=50,
        min_rr_ratio=1.5,
        volatility_factor=1.0
    )
    
    labels_train = label_builder.build_labels(klines_train).reset_index(drop=True)
    labels_backtest = label_builder.build_labels(klines_backtest).reset_index(drop=True)
    
    # 数据对齐
    min_len_train = min(len(X_train_full), len(labels_train))
    X_train_full = X_train_full.iloc[:min_len_train]
    labels_train = labels_train.iloc[:min_len_train]
    
    min_len_backtest = min(len(X_backtest_full), len(labels_backtest), len(klines_backtest))
    X_backtest_full = X_backtest_full.iloc[:min_len_backtest]
    labels_backtest = labels_backtest.iloc[:min_len_backtest]
    klines_backtest = klines_backtest.iloc[:min_len_backtest]
    
    # 第1轮训练：用全部特征获取特征重要性
    logger.info("\n【步骤5】第1轮训练（全部特征，获取特征重要性）")
    split_idx = int(len(X_train_full) * 0.8)
    X_train_split = X_train_full.iloc[:split_idx].reset_index(drop=True)
    X_val = X_train_full.iloc[split_idx:].reset_index(drop=True)
    labels_train_split = labels_train.iloc[:split_idx].reset_index(drop=True)
    labels_val = labels_train.iloc[split_idx:].reset_index(drop=True)
    
    strategy_full = TwoStageRiskRewardStrategy(loss_function='rmse')
    metrics = strategy_full.train(X_train_split, labels_train_split, X_val, labels_val, rr_threshold=2.0)
    
    # 获取Top-30特征
    top_30_features = get_top_30_features(strategy_full)
    
    # 筛选Top-30特征
    logger.info("\n【步骤6】筛选Top-30特征")
    X_train_top30 = X_train_full[top_30_features]
    X_backtest_top30 = X_backtest_full[top_30_features]
    
    X_train_split_top30 = X_train_top30.iloc[:split_idx].reset_index(drop=True)
    X_val_top30 = X_train_top30.iloc[split_idx:].reset_index(drop=True)
    
    # 第2轮训练：用Top-30特征训练最终模型
    logger.info("\n【步骤7】第2轮训练（Top-30特征）")
    strategy = TwoStageRiskRewardStrategy(loss_function='rmse')
    metrics = strategy.train(X_train_split_top30, labels_train_split, X_val_top30, labels_val, rr_threshold=2.0)
    
    # 保存模型
    logger.info("\n【步骤8】保存模型")
    model_dir = Path('models/optimized_rr_strategy_top30')
    strategy.save(model_dir)
    
    # 保存特征列表
    feature_list_path = model_dir / 'top30_features.txt'
    with open(feature_list_path, 'w') as f:
        for feat in top_30_features:
            f.write(f"{feat}\n")
    logger.info(f"Top-30特征列表已保存: {feature_list_path}")
    
    # 预测回测集
    logger.info("\n【步骤9】预测回测集信号")
    predictions = strategy.predict(X_backtest_top30, rr_threshold=2.0, prob_threshold=0.65)
    
    logger.info(f"总样本: {len(predictions):,}")
    logger.info(f"应交易样本: {predictions['should_trade'].sum():,} ({predictions['should_trade'].sum()/len(predictions)*100:.1f}%)")
    
    # 回测
    logger.info("\n【步骤10】回测盈亏比驱动策略（简化回测逻辑）")
    results = simple_backtest(klines_backtest, predictions, initial_balance=1000.0, position_size_pct=0.10)
    
    # 打印回测结果
    logger.info("\n" + "=" * 100)
    logger.info("【回测结果】优化后的盈亏比驱动策略")
    logger.info("=" * 100)
    
    logger.info(f"\n核心指标:")
    logger.info(f"  总收益率: {results['total_return']:.2f}%")
    logger.info(f"  胜率: {results['win_rate']:.2f}%")
    logger.info(f"  总交易次数: {results['total_trades']}")
    logger.info(f"  盈利交易: {results['winning_trades']}")
    logger.info(f"  亏损交易: {results['losing_trades']}")
    
    logger.info(f"\n盈亏分析:")
    logger.info(f"  平均盈利: {results['avg_profit']:.2f}")
    logger.info(f"  平均亏损: {results['avg_loss']:.2f}")
    logger.info(f"  盈亏比: {results['profit_loss_ratio']:.2f}")
    
    # 对比基准
    benchmark_return = (klines_backtest['close'].iloc[-1] / klines_backtest['close'].iloc[0] - 1) * 100
    logger.info(f"\n基准（买入持有）:")
    logger.info(f"  收益率: {benchmark_return:.2f}%")
    logger.info(f"  超额收益: {(results['total_return'] - benchmark_return):.2f}%")
    
    # 保存交易记录
    if results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        Path('backtest').mkdir(exist_ok=True)
        output_file = f"backtest/optimized_top30_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(output_file, index=False)
        logger.info(f"\n交易记录已保存: {output_file}")
    
    logger.info("\n" + "=" * 100)
    logger.info("训练和回测完成！")
    logger.info("=" * 100)
    
    # 返回关键指标
    return {
        'train_metrics': metrics,
        'backtest_results': results,
        'top_30_features': top_30_features
    }


if __name__ == '__main__':
    try:
        results = main()
        
        # 打印最终总结
        print("\n" + "=" * 100)
        print("【最终总结】")
        print("=" * 100)
        print(f"配置: Top-30特征 + RR=2.0 + RMSE + 简化回测")
        print(f"胜率: {results['backtest_results']['win_rate']:.2f}%")
        print(f"总收益率: {results['backtest_results']['total_return']:.2f}%")
        print(f"盈亏比: {results['backtest_results']['profit_loss_ratio']:.2f}")
        print(f"交易次数: {results['backtest_results']['total_trades']}")
        
        if results['backtest_results']['total_return'] >= 120.0:
            print("\n✅ 目标达成！收益率超过120%")
        
        print("=" * 100)
        
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        sys.exit(1)
