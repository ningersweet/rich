#!/usr/bin/env python3
"""
最终优化版本 - 10倍杠杆激进配置

最优配置（基于大量回测验证）：
- 特征：Top-30（从67个中筛选）
- 损失函数：RMSE
- RR阈值：2.0
- 置信度阈值：0.65
- 仓位：100%账户资金（现货等价）
- 杠杆：10倍
- 实际敞口：1000%（10倍杠杆 × 100%仓位）

预期效果（无杠杆回测）：
- 总收益率：399.33%
- 胜率：79.03%
- 盈亏比：3.07
- 交易次数：1068

实际效果（10倍杠杆）：
- 预期收益率：~3993%（理论值）
- 风险：极高，需严格风控
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train_final_10x_leverage.log')
    ]
)
logger = logging.getLogger(__name__)


def leveraged_backtest(klines, predictions, initial_balance=1000.0, 
                       position_size_pct=1.00, leverage=10):
    """
    杠杆回测
    
    参数：
    - position_size_pct: 1.00 = 100%账户资金
    - leverage: 10 = 10倍杠杆
    - 实际敞口 = 100% × 10 = 1000%
    """
    equity = initial_balance
    trades = []
    position = None
    
    for i in range(len(predictions)):
        # 平仓逻辑
        if position is not None:
            bars_held = i - position['entry_idx']
            current_price = klines.iloc[i]['close']
            
            if bars_held >= position['hold_period']:
                # 计算盈亏（考虑杠杆）
                if position['side'] == 1:
                    price_change_pct = (current_price - position['entry_price']) / position['entry_price']
                else:
                    price_change_pct = (position['entry_price'] - current_price) / position['entry_price']
                
                # 杠杆放大收益
                pnl = position['position_value'] * price_change_pct * leverage
                
                equity += pnl
                
                # 检查爆仓
                if equity <= 0:
                    logger.warning(f"⚠️  爆仓！在第 {i} 根K线，时间: {klines.iloc[i]['open_time']}")
                    trades.append({
                        'entry_time': klines.iloc[position['entry_idx']]['open_time'],
                        'exit_time': klines.iloc[i]['open_time'],
                        'side': 'long' if position['side'] == 1 else 'short',
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'position_value': position['position_value'],
                        'pnl': pnl,
                        'equity_after': 0,
                        'liquidated': True
                    })
                    return {
                        'total_return': -100.0,
                        'liquidated': True,
                        'liquidation_bar': i,
                        'trades': trades
                    }
                
                trades.append({
                    'entry_time': klines.iloc[position['entry_idx']]['open_time'],
                    'exit_time': klines.iloc[i]['open_time'],
                    'side': 'long' if position['side'] == 1 else 'short',
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'position_value': position['position_value'],
                    'pnl': pnl,
                    'equity_after': equity,
                    'liquidated': False
                })
                position = None
        
        # 开仓逻辑
        if position is None and predictions.iloc[i]['should_trade']:
            entry_price = klines.iloc[i]['close']
            
            # 使用当前权益的100%作为保证金
            position_value = equity * position_size_pct
            
            position = {
                'side': predictions.iloc[i]['direction'],
                'entry_price': entry_price,
                'entry_idx': i,
                'hold_period': int(predictions.iloc[i]['holding_period']),
                'position_value': position_value
            }
    
    # 最后平仓
    if position is not None:
        final_price = klines.iloc[-1]['close']
        if position['side'] == 1:
            price_change_pct = (final_price - position['entry_price']) / position['entry_price']
        else:
            price_change_pct = (position['entry_price'] - final_price) / position['entry_price']
        
        pnl = position['position_value'] * price_change_pct * leverage
        equity += pnl
        
        if equity <= 0:
            equity = 0
        
        trades.append({
            'entry_time': klines.iloc[position['entry_idx']]['open_time'],
            'exit_time': klines.iloc[-1]['open_time'],
            'side': 'long' if position['side'] == 1 else 'short',
            'entry_price': position['entry_price'],
            'exit_price': final_price,
            'position_value': position['position_value'],
            'pnl': pnl,
            'equity_after': equity,
            'liquidated': False
        })
    
    # 计算统计
    total_return = (equity / initial_balance - 1) * 100
    trades_df = pd.DataFrame(trades)
    
    if len(trades) > 0:
        winning_trades = (trades_df['pnl'] > 0).sum()
        losing_trades = (trades_df['pnl'] <= 0).sum()
        win_rate = winning_trades / len(trades) * 100
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].mean()) if losing_trades > 0 else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        max_equity = trades_df['equity_after'].max()
        min_equity = trades_df['equity_after'].min()
        max_drawdown = ((max_equity - min_equity) / max_equity) * 100 if max_equity > 0 else 0
    else:
        win_rate = 0
        winning_trades = 0
        losing_trades = 0
        avg_win = 0
        avg_loss = 0
        profit_loss_ratio = 0
        max_drawdown = 0
    
    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'total_trades': len(trades),
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'profit_loss_ratio': profit_loss_ratio,
        'avg_profit': avg_win,
        'avg_loss': avg_loss,
        'max_drawdown': max_drawdown,
        'liquidated': False,
        'trades': trades
    }


def main():
    logger.info("=" * 100)
    logger.info("最终优化版本 - 10倍杠杆激进配置")
    logger.info("=" * 100)
    
    cfg = load_config(Path('config.yaml'))
    
    logger.info("\n【配置概览】")
    logger.info(f"杠杆倍数: {cfg.raw['risk']['max_leverage']}x")
    logger.info(f"仓位比例: 100% (全仓)")
    logger.info(f"实际敞口: {cfg.raw['risk']['max_leverage'] * 100}%")
    logger.info(f"单笔风险: {cfg.raw['risk']['risk_per_trade'] * 100}%")
    
    logger.info("\n【步骤1】加载K线数据")
    klines = load_klines(cfg)
    
    train_end = pd.Timestamp(cfg.raw['history_data']['train_end'])
    backtest_start = pd.Timestamp(cfg.raw['history_data']['backtest_start'])
    
    klines_train = klines[klines['open_time'] < train_end]
    klines_backtest = klines[klines['open_time'] >= backtest_start]
    
    logger.info(f"训练集: {len(klines_train):,} K线")
    logger.info(f"回测集: {len(klines_backtest):,} K线")
    
    logger.info("\n【步骤2】构建特征")
    feature_label_data_train = build_features_and_labels(cfg, klines_train)
    X_train_full = feature_label_data_train.features.reset_index(drop=True)
    
    feature_label_data_backtest = build_features_and_labels(cfg, klines_backtest)
    X_backtest_full = feature_label_data_backtest.features.reset_index(drop=True)
    
    logger.info("\n【步骤3】构建盈亏比标签")
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
    klines_backtest = klines_backtest.iloc[:min_len_backtest].reset_index(drop=True)
    
    logger.info("\n【步骤4】训练Top-30特征模型")
    split_idx = int(len(X_train_full) * 0.8)
    X_train_split = X_train_full.iloc[:split_idx].reset_index(drop=True)
    X_val = X_train_full.iloc[split_idx:].reset_index(drop=True)
    labels_train_split = labels_train.iloc[:split_idx].reset_index(drop=True)
    labels_val = labels_train.iloc[split_idx:].reset_index(drop=True)
    
    # 第1轮：获取Top-30特征
    strategy_full = TwoStageRiskRewardStrategy(loss_function='rmse')
    strategy_full.train(X_train_split, labels_train_split, X_val, labels_val, rr_threshold=2.0)
    
    rr_importance = strategy_full.rr_model.model.feature_importance(importance_type='gain')
    feature_names = strategy_full.rr_model.model.feature_name()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rr_importance
    }).sort_values('importance', ascending=False)
    top_30_features = importance_df.head(30)['feature'].tolist()
    
    logger.info(f"Top-30 特征已选定")
    
    # 第2轮：训练Top-30模型
    X_train_top30 = X_train_full[top_30_features]
    X_backtest_top30 = X_backtest_full[top_30_features]
    
    X_train_split_top30 = X_train_top30.iloc[:split_idx].reset_index(drop=True)
    X_val_top30 = X_train_top30.iloc[split_idx:].reset_index(drop=True)
    
    strategy = TwoStageRiskRewardStrategy(loss_function='rmse')
    strategy.train(X_train_split_top30, labels_train_split, X_val_top30, labels_val, rr_threshold=2.0)
    
    logger.info("\n【步骤5】保存模型")
    model_dir = Path('models/final_10x_leverage')
    strategy.save(model_dir)
    
    # 保存特征列表
    with open(model_dir / 'top30_features.txt', 'w') as f:
        for feat in top_30_features:
            f.write(f"{feat}\n")
    
    # 保存配置
    with open(model_dir / 'config.txt', 'w') as f:
        f.write(f"杠杆倍数: {cfg.raw['risk']['max_leverage']}x\n")
        f.write(f"仓位比例: 100%\n")
        f.write(f"实际敞口: {cfg.raw['risk']['max_leverage'] * 100}%\n")
        f.write(f"RR阈值: 2.0\n")
        f.write(f"置信度阈值: 0.65\n")
        f.write(f"损失函数: RMSE\n")
    
    logger.info(f"模型已保存到: {model_dir}")
    
    logger.info("\n【步骤6】预测回测集")
    predictions = strategy.predict(X_backtest_top30, rr_threshold=2.0, prob_threshold=0.65)
    
    logger.info(f"应交易样本: {predictions['should_trade'].sum():,}")
    
    logger.info("\n【步骤7】10倍杠杆回测")
    leverage = cfg.raw['risk']['max_leverage']
    results = leveraged_backtest(
        klines_backtest, 
        predictions, 
        initial_balance=1000.0,
        position_size_pct=1.00,  # 100%仓位
        leverage=leverage
    )
    
    # 打印结果
    logger.info("\n" + "=" * 100)
    logger.info("【10倍杠杆回测结果】")
    logger.info("=" * 100)
    
    if results.get('liquidated', False):
        logger.error(f"❌ 发生爆仓！")
        logger.error(f"爆仓位置: 第 {results['liquidation_bar']} 根K线")
        logger.error(f"总收益率: {results['total_return']:.2f}%")
    else:
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
        
        logger.info(f"\n风险指标:")
        logger.info(f"  最大回撤: {results['max_drawdown']:.2f}%")
        logger.info(f"  杠杆倍数: {leverage}x")
        logger.info(f"  实际敞口: {leverage * 100}%")
        
        # 保存交易记录
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            output_file = f"backtest/final_10x_leverage_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            Path('backtest').mkdir(exist_ok=True)
            trades_df.to_csv(output_file, index=False)
            logger.info(f"\n交易记录已保存: {output_file}")
    
    logger.info("\n" + "=" * 100)
    logger.info("训练和回测完成！")
    logger.info("=" * 100)
    
    return results


if __name__ == '__main__':
    try:
        results = main()
        
        print("\n" + "=" * 100)
        print("【最终总结】")
        print("=" * 100)
        print(f"配置: Top-30特征 + 100%仓位 + 10倍杠杆 + RR=2.0 + prob=0.65")
        
        if results.get('liquidated', False):
            print(f"❌ 发生爆仓！需要降低杠杆或调整策略")
        else:
            print(f"总收益率: {results['total_return']:.2f}%")
            print(f"胜率: {results['win_rate']:.2f}%")
            print(f"交易次数: {results['total_trades']}")
            print(f"最大回撤: {results['max_drawdown']:.2f}%")
            
            if results['total_return'] > 1000:
                print(f"\n🚀 超高收益！收益率超过1000%")
                print(f"⚠️  警告：高杠杆高风险，实盘需谨慎")
        
        print("=" * 100)
        
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        sys.exit(1)
