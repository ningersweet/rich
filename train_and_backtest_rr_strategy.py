#!/usr/bin/env python3
"""
盈亏比驱动策略：完整训练和回测流程

目标：胜率45%+
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
from btc_quant.backtest import run_backtest_with_triple_exit

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rr_strategy_training.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """主流程"""
    logger.info("=" * 100)
    logger.info("盈亏比驱动策略 - 训练与回测")
    logger.info("=" * 100)
    
    # 加载配置
    cfg_path = Path(__file__).parent / 'config.yaml'
    cfg = load_config(cfg_path)
    
    # 加载K线数据
    logger.info("\n【步骤1】加载K线数据")
    klines = load_klines(cfg)
    logger.info(f"总K线数: {len(klines):,}")
    logger.info(f"时间范围: {klines.index[0]} ~ {klines.index[-1]}")
    
    # 划分数据集
    logger.info("\n【步骤2】划分数据集")
    train_end = pd.Timestamp(cfg.raw['history_data']['train_end'])
    backtest_start = pd.Timestamp(cfg.raw['history_data']['backtest_start'])
    
    # 使用open_time列进行划分
    klines_train = klines[klines['open_time'] < train_end]
    klines_backtest = klines[klines['open_time'] >= backtest_start]
    
    logger.info(f"训练集: {len(klines_train):,} ({klines_train.index[0]} ~ {klines_train.index[-1]})")
    logger.info(f"回测集: {len(klines_backtest):,} ({klines_backtest.index[0]} ~ {klines_backtest.index[-1]})")
    
    # 构建特征
    logger.info("\n【步骤3】构建特征")
    feature_label_data_train = build_features_and_labels(cfg, klines_train)
    X_train = feature_label_data_train.features
    
    feature_label_data_backtest = build_features_and_labels(cfg, klines_backtest)
    X_backtest = feature_label_data_backtest.features
    
    # 重置index以便后续对齐
    X_train = X_train.reset_index(drop=True)
    X_backtest = X_backtest.reset_index(drop=True)
    
    logger.info(f"训练特征: {X_train.shape}")
    logger.info(f"回测特征: {X_backtest.shape}")
    logger.info(f"特征列表: {list(X_train.columns[:10])}... (共{len(X_train.columns)}个)")
    
    # 构建盈亏比标签
    logger.info("\n【步骤4】构建盈亏比标签")
    label_builder = RiskRewardLabelBuilder(
        target_return=0.01,          # 1%目标收益
        max_holding_period=50,        # 观察未来50根K线
        min_rr_ratio=1.5,            # 最小盈亏比1.5
        volatility_factor=1.0
    )
    
    logger.info("构建训练集标签...")
    labels_train = label_builder.build_labels(klines_train)
    labels_train = labels_train.reset_index(drop=True)  # 重置index
    
    logger.info("构建回测集标签...")
    labels_backtest = label_builder.build_labels(klines_backtest)
    labels_backtest = labels_backtest.reset_index(drop=True)  # 重置index
    
    # 同时重置klines的index
    klines_train = klines_train.reset_index(drop=True)
    klines_backtest = klines_backtest.reset_index(drop=True)
    
    stats_train = label_builder.get_label_statistics(labels_train)
    stats_backtest = label_builder.get_label_statistics(labels_backtest)
    
    logger.info(f"\n训练集统计:")
    logger.info(f"  交易样本占比: {stats_train['active_ratio']*100:.1f}%")
    logger.info(f"  平均盈亏比: {stats_train.get('avg_risk_reward', 0):.2f}")
    logger.info(f"  高质量交易(RR>2.0): {stats_train.get('high_rr_ratio', 0)*100:.1f}%")
    
    logger.info(f"\n回测集统计:")
    logger.info(f"  交易样本占比: {stats_backtest['active_ratio']*100:.1f}%")
    logger.info(f"  平均盈亏比: {stats_backtest.get('avg_risk_reward', 0):.2f}")
    logger.info(f"  高质量交易(RR>2.0): {stats_backtest.get('high_rr_ratio', 0)*100:.1f}%")
    
    # 调试信息：检查index
    logger.info(f"\n调试信息 - 数据结构:")
    logger.info(f"  X_train index: {X_train.index[:5].tolist()} ... {X_train.index[-5:].tolist()}")
    logger.info(f"  labels_train index: {labels_train.index[:5].tolist()} ... {labels_train.index[-5:].tolist()}")
    logger.info(f"  X_backtest index: {X_backtest.index[:5].tolist()} ... {X_backtest.index[-5:].tolist()}")
    logger.info(f"  labels_backtest index: {labels_backtest.index[:5].tolist()} ... {labels_backtest.index[-5:].tolist()}")
    logger.info(f"  klines_backtest index: {klines_backtest.index[:5].tolist()} ... {klines_backtest.index[-5:].tolist()}")
    
    # 对齐数据（去除NaN）
    # 现在index已统一，直接过滤NaN
    valid_idx_train = X_train.notna().all(axis=1) & labels_train.notna().all(axis=1)
    X_train_clean = X_train[valid_idx_train].copy()
    labels_train_clean = labels_train[valid_idx_train].copy()
    
    valid_idx_backtest = X_backtest.notna().all(axis=1) & labels_backtest.notna().all(axis=1)
    X_backtest_clean = X_backtest[valid_idx_backtest].copy()
    labels_backtest_clean = labels_backtest[valid_idx_backtest].copy()
    klines_backtest_clean = klines_backtest[valid_idx_backtest].copy()
    
    logger.info(f"\n数据对齐后:")
    logger.info(f"  训练集: {len(X_train_clean):,}")
    logger.info(f"  回测集: {len(X_backtest_clean):,}")
    
    # 训练两阶段模型
    logger.info("\n【步骤5】训练两阶段模型")
    
    # 使用训练集的后20%作为验证集
    val_size = int(len(X_train_clean) * 0.2)
    X_train_split = X_train_clean.iloc[:-val_size]
    labels_train_split = labels_train_clean.iloc[:-val_size]
    X_val_split = X_train_clean.iloc[-val_size:]
    labels_val_split = labels_train_clean.iloc[-val_size:]
    
    logger.info(f"训练集: {len(X_train_split):,}, 验证集: {len(X_val_split):,}")
    
    strategy = TwoStageRiskRewardStrategy()
    
    metrics = strategy.train(
        X_train_split, labels_train_split,
        X_val_split, labels_val_split,
        rr_threshold=2.0
    )
    
    # 保存模型
    model_dir = Path('models/risk_reward_strategy')
    strategy.save(model_dir)
    
    # 预测回测集
    logger.info("\n【步骤6】预测回测集信号")
    predictions = strategy.predict(
        X_backtest_clean,
        rr_threshold=2.0,
        prob_threshold=0.65
    )
    
    # 统计预测信号
    signal_stats = {
        '总样本': len(predictions),
        '高RR样本(>2.0)': (predictions['predicted_rr'] > 2.0).sum(),
        '做多信号': (predictions['direction'] == 1).sum(),
        '做空信号': (predictions['direction'] == -1).sum(),
        '观望信号': (predictions['direction'] == 0).sum(),
        '应交易样本': predictions['should_trade'].sum(),
    }
    
    logger.info("\n预测信号统计:")
    for key, value in signal_stats.items():
        pct = value / len(predictions) * 100 if len(predictions) > 0 else 0
        logger.info(f"  {key}: {value:,} ({pct:.1f}%)")
    
    # 回测
    logger.info("\n【步骤7】回测盈亏比驱动策略")
    
    # 准备回测数据
    # 将预测结果转换为信号格式
    signals = pd.Series(0, index=predictions.index)
    signals[predictions['should_trade'] & (predictions['direction'] == 1)] = 1   # 做多
    signals[predictions['should_trade'] & (predictions['direction'] == -1)] = -1  # 做空
    
    # 位置比率（暂时固定为1，后续可根据盈亏比调整）
    position_ratios = pd.Series(1.0, index=predictions.index)
    
    # 执行回测：使用预测的持有周期和动态止损
    logger.info("使用预测的持有周期和盈亏比执行回测...")
    
    # 简化回测逻辑：按预测的持有周期执行
    initial_balance = cfg.raw['backtest'].get('initial_balance', 1000.0)
    equity = initial_balance
    trades = []
    position = None  # {'side': 1/-1, 'entry_price': float, 'entry_idx': int, 'hold_period': int}
    
    for i, row in predictions.iterrows():
        # 如果有持仓，检查是否应该出场
        if position is not None:
            bars_held = i - position['entry_idx']
            current_price = klines_backtest_clean.loc[i, 'close']
            
            # 达到预期持有周期，平仓
            if bars_held >= position['hold_period']:
                if position['side'] == 1:  # 多头
                    pnl = (current_price - position['entry_price']) * position['quantity']
                else:  # 空头
                    pnl = (position['entry_price'] - current_price) * position['quantity']
                
                equity += pnl
                trades.append({
                    'entry_time': klines_backtest_clean.loc[position['entry_idx'], 'open_time'],
                    'exit_time': klines_backtest_clean.loc[i, 'open_time'],
                    'side': 'long' if position['side'] == 1 else 'short',
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'quantity': position['quantity'],
                    'pnl': pnl
                })
                position = None
        
        # 如果无持仓且有交易信号，开仓
        if position is None and row['should_trade']:
            entry_price = klines_backtest_clean.loc[i, 'close']
            quantity = (equity * 0.01) / entry_price  # 1%仓位
            
            position = {
                'side': row['direction'],
                'entry_price': entry_price,
                'entry_idx': i,
                'hold_period': int(row['holding_period']),
                'quantity': quantity
            }
    
    # 最后如果还有持仓，平仓
    if position is not None:
        final_price = klines_backtest_clean.iloc[-1]['close']
        if position['side'] == 1:
            pnl = (final_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - final_price) * position['quantity']
        
        equity += pnl
        trades.append({
            'entry_time': klines_backtest_clean.loc[position['entry_idx'], 'open_time'],
            'exit_time': klines_backtest_clean.iloc[-1]['open_time'],
            'side': 'long' if position['side'] == 1 else 'short',
            'entry_price': position['entry_price'],
            'exit_price': final_price,
            'quantity': position['quantity'],
            'pnl': pnl
        })
    
    # 计算统计指标
    total_return = (equity / initial_balance - 1) * 100
    
    trades_df = pd.DataFrame(trades)
    winning_trades = (trades_df['pnl'] > 0).sum()
    losing_trades = (trades_df['pnl'] <= 0).sum()
    win_rate = winning_trades / len(trades) * 100 if len(trades) > 0 else 0
    
    results = {
        'total_return': total_return,
        'win_rate': win_rate,
        'total_trades': len(trades),
        'trades': trades,
        'final_equity': equity
    }
    
    # 打印回测结果
    logger.info("\n" + "=" * 100)
    logger.info("【回测结果】盈亏比驱动策略")
    logger.info("=" * 100)
    
    logger.info(f"\n核心指标:")
    logger.info(f"  总收益率: {results['total_return']:.2f}%")
    logger.info(f"  胜率: {results['win_rate']:.2f}%")
    logger.info(f"  总交易次数: {results['total_trades']}")
    
    # 计算盈亏交易数
    if 'trades' in results and len(results['trades']) > 0:
        trades_df = pd.DataFrame(results['trades'])
        winning_trades = (trades_df['pnl'] > 0).sum()
        losing_trades = (trades_df['pnl'] <= 0).sum()
        
        logger.info(f"  盈利交易: {winning_trades}")
        logger.info(f"  亏损交易: {losing_trades}")
        
        # 盈亏分析
        if winning_trades > 0 and losing_trades > 0:
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
            avg_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].mean())
            
            logger.info(f"\n盈亏分析:")
            logger.info(f"  平均盈利: {avg_win:.2f}")
            logger.info(f"  平均亏损: {avg_loss:.2f}")
            logger.info(f"  盈亏比: {avg_win/avg_loss:.2f}")
    
    # 对比基准（买入持有）
    benchmark_return = (klines_backtest_clean['close'].iloc[-1] / klines_backtest_clean['close'].iloc[0] - 1) * 100
    logger.info(f"\n基准（买入持有）:")
    logger.info(f"  收益率: {benchmark_return:.2f}%")
    logger.info(f"  超额收益: {(results['total_return'] - benchmark_return):.2f}%")
    
    # 保存详细交易记录
    if 'trades' in results and len(results['trades']) > 0:
        trades_df = pd.DataFrame(results['trades'])
        # 确保backtest目录存在
        Path('backtest').mkdir(exist_ok=True)
        output_file = f"backtest/backtest_results_rr_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(output_file, index=False)
        logger.info(f"\n交易记录已保存: {output_file}")
        
        # 显示前10笔交易
        logger.info("\n前10笔交易:")
        print(trades_df.head(10).to_string())
    
    logger.info("\n" + "=" * 100)
    logger.info("训练和回测完成！")
    logger.info("=" * 100)
    
    # 返回关键指标
    return {
        'train_metrics': metrics,
        'backtest_results': results,
        'signal_stats': signal_stats,
    }


if __name__ == '__main__':
    try:
        results = main()
        
        # 打印最终总结
        print("\n" + "=" * 100)
        print("【最终总结】")
        print("=" * 100)
        print(f"胜率: {results['backtest_results']['win_rate']:.2f}% (目标: 45%)")
        print(f"总收益率: {results['backtest_results']['total_return']:.2f}%")
        print(f"交易次数: {results['backtest_results']['total_trades']}")
        
        if results['backtest_results']['win_rate'] >= 45.0:
            print("\n✅ 目标达成！胜率超过45%")
        else:
            print(f"\n⚠️  胜率未达标，需要继续优化（当前: {results['backtest_results']['win_rate']:.1f}%）")
        
        print("=" * 100)
        
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        sys.exit(1)
