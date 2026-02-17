#!/usr/bin/env python3
"""
凯利公式仓位管理回测对比
对比三种敞口策略：
1. 固定敞口（当前方案）
2. 动态敞口（基于RR和Prob的线性规则）
3. 凯利公式（科学计算最优仓位）
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from btc_quant.config import load_config
from btc_quant.data import load_klines
from btc_quant.features import build_features_and_labels
from btc_quant.risk_reward_model import TwoStageRiskRewardStrategy
from backtest_atr_stop_loss import logger
from btc_quant.kelly_position import KellyPositionManager
from run_live_dynamic_exposure import calculate_dynamic_exposure


def backtest_with_fixed_exposure(klines, predictions, initial_balance=1000.0, fixed_exposure=5.0):
    """
    固定敞口回测（简化版，用于对比）
    """
    equity = initial_balance
    trades = []
    position = None
    
    for i in range(len(predictions)):
        current_price = klines.iloc[i]['close']
        
        if position is None and predictions.iloc[i]['should_trade']:
            entry_price = current_price
            direction = predictions.iloc[i]['direction']
            hold_period = int(predictions.iloc[i]['holding_period'])
            
            position = {
                'side': direction,
                'entry_price': entry_price,
                'entry_idx': i,
                'exposure': fixed_exposure,
                'hold_period': hold_period
            }
        
        elif position is not None:
            bars_held = i - position['entry_idx']
            
            if bars_held >= position['hold_period']:
                if position['side'] == 1:
                    price_change_pct = (current_price - position['entry_price']) / position['entry_price']
                else:
                    price_change_pct = (position['entry_price'] - current_price) / position['entry_price']
                
                pnl = initial_balance * price_change_pct * position['exposure']
                equity += pnl
                
                trades.append({
                    'entry_time': klines.iloc[position['entry_idx']]['open_time'],
                    'exit_time': klines.iloc[i]['open_time'],
                    'side': 'long' if position['side'] == 1 else 'short',
                    'pnl': pnl,
                    'pnl_pct': price_change_pct * 100 * position['exposure'],
                    'equity_after': equity,
                    'exposure_used': position['exposure']
                })
                
                position = None
    
    total_return = (equity - initial_balance) / initial_balance * 100
    win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100 if trades else 0
    
    return {
        'strategy': f'固定敞口×{fixed_exposure}',
        'initial_balance': initial_balance,
        'final_equity': equity,
        'total_return': total_return,
        'win_rate': win_rate,
        'total_trades': len(trades),
        'trades': trades
    }


def backtest_with_dynamic_exposure(klines, predictions, initial_balance=1000.0, max_exposure=10.0):
    """
    动态敞口回测（当前方案：基于RR和Prob的线性规则）
    """
    equity = initial_balance
    trades = []
    position = None
    
    for i in range(len(predictions)):
        current_price = klines.iloc[i]['close']
        
        if position is None and predictions.iloc[i]['should_trade']:
            entry_price = current_price
            predicted_rr = predictions.iloc[i]['predicted_rr']
            direction_prob = predictions.iloc[i]['direction_prob']
            direction = predictions.iloc[i]['direction']
            hold_period = int(predictions.iloc[i]['holding_period'])
            
            # 使用现有动态敞口公式
            exposure = calculate_dynamic_exposure(
                predicted_rr=predicted_rr,
                direction_prob=direction_prob,
                max_exposure=max_exposure
            )
            
            position = {
                'side': direction,
                'entry_price': entry_price,
                'entry_idx': i,
                'exposure': exposure,
                'hold_period': hold_period
            }
        
        elif position is not None:
            bars_held = i - position['entry_idx']
            
            if bars_held >= position['hold_period']:
                if position['side'] == 1:
                    price_change_pct = (current_price - position['entry_price']) / position['entry_price']
                else:
                    price_change_pct = (position['entry_price'] - current_price) / position['entry_price']
                
                pnl = initial_balance * price_change_pct * position['exposure']
                equity += pnl
                
                trades.append({
                    'entry_time': klines.iloc[position['entry_idx']]['open_time'],
                    'exit_time': klines.iloc[i]['open_time'],
                    'side': 'long' if position['side'] == 1 else 'short',
                    'pnl': pnl,
                    'pnl_pct': price_change_pct * 100 * position['exposure'],
                    'equity_after': equity,
                    'exposure_used': position['exposure']
                })
                
                position = None
    
    total_return = (equity - initial_balance) / initial_balance * 100
    win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100 if trades else 0
    
    return {
        'strategy': '动态敞口（线性规则）',
        'initial_balance': initial_balance,
        'final_equity': equity,
        'total_return': total_return,
        'win_rate': win_rate,
        'total_trades': len(trades),
        'trades': trades
    }


def backtest_with_kelly_exposure(klines, predictions, initial_balance=1000.0, 
                                  kelly_criterion=0.5, max_exposure=10.0, risk_mode='balanced'):
    """
    凯利公式敞口回测
    """
    position_manager = KellyPositionManager(
        kelly_criterion=kelly_criterion,
        max_exposure=max_exposure,
        risk_mode=risk_mode
    )
    
    equity = initial_balance
    trades = []
    position = None
    
    for i in range(len(predictions)):
        current_price = klines.iloc[i]['close']
        
        if position is None and predictions.iloc[i]['should_trade']:
            entry_price = current_price
            predicted_rr = predictions.iloc[i]['predicted_rr']
            direction_prob = predictions.iloc[i]['direction_prob']
            direction = predictions.iloc[i]['direction']
            hold_period = int(predictions.iloc[i]['holding_period'])
            
            # 使用凯利公式计算敞口
            exposure, reasoning = position_manager.calculate_optimal_exposure(
                predicted_rr=predicted_rr,
                direction_prob=direction_prob
            )
            
            # 如果信号质量太差，跳过交易
            if exposure < 1.0:
                continue
            
            position = {
                'side': direction,
                'entry_price': entry_price,
                'entry_idx': i,
                'exposure': exposure,
                'hold_period': hold_period
            }
        
        elif position is not None:
            bars_held = i - position['entry_idx']
            
            if bars_held >= position['hold_period']:
                if position['side'] == 1:
                    price_change_pct = (current_price - position['entry_price']) / position['entry_price']
                else:
                    price_change_pct = (position['entry_price'] - current_price) / position['entry_price']
                
                pnl = initial_balance * price_change_pct * position['exposure']
                equity += pnl
                
                # 更新凯利管理器
                position_manager.update_equity(equity, pnl)
                
                trades.append({
                    'entry_time': klines.iloc[position['entry_idx']]['open_time'],
                    'exit_time': klines.iloc[i]['open_time'],
                    'side': 'long' if position['side'] == 1 else 'short',
                    'pnl': pnl,
                    'pnl_pct': price_change_pct * 100 * position['exposure'],
                    'equity_after': equity,
                    'exposure_used': position['exposure']
                })
                
                position = None
    
    total_return = (equity - initial_balance) / initial_balance * 100
    win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100 if trades else 0
    
    # 计算额外指标
    import pandas as pd
    equity_curve = [initial_balance]
    for t in trades:
        equity_curve.append(t['equity_after'])
    equity_series = pd.Series(equity_curve)
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = drawdown.min()
    
    returns = equity_series.pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 96) if returns.std() > 0 else 0
    
    return {
        'strategy': f'凯利公式 ({risk_mode})',
        'initial_balance': initial_balance,
        'final_equity': equity,
        'total_return': total_return,
        'win_rate': win_rate,
        'total_trades': len(trades),
        'trades': trades,
        'max_drawdown': max_drawdown * 100,
        'sharpe_ratio': sharpe_ratio
    }


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("凯利公式仓位管理回测对比")
    logger.info("="*60)
    
    # 1. 加载配置和数据
    cfg = load_config(Path('config.yaml'))
    logger.info("✅ 配置加载完成")
    
    klines = load_klines(cfg)
    logger.info(f"✅ K线加载完成，共 {len(klines)} 根")
    
    # 2. 构建特征
    logger.info("构建特征...")
    feature_label_data = build_features_and_labels(cfg, klines)
    X_full = feature_label_data.features.reset_index(drop=True)
    
    min_len = min(len(X_full), len(klines))
    X_full = X_full.iloc[:min_len]
    klines = klines.iloc[:min_len].reset_index(drop=True)
    
    if 'atr_14' in X_full.columns:
        klines['atr_14'] = X_full['atr_14'].values
    else:
        raise ValueError("特征中缺少 atr_14 列")
    
    logger.info(f"✅ 特征构建完成，样本数：{min_len}")
    
    # 3. 筛选样本外数据
    sample_out_mask = klines['open_time'] >= pd.Timestamp('2025-01-01', tz='UTC')
    klines_test = klines[sample_out_mask].reset_index(drop=True)
    X_test = X_full[sample_out_mask].reset_index(drop=True)
    logger.info(f"✅ 样本外数据筛选完成：{len(klines_test)}根K线")
    
    # 4. 加载模型
    model_dir = Path('models/final_6x_fixed_capital')
    strategy = TwoStageRiskRewardStrategy()
    strategy.load(model_dir)
    logger.info(f"✅ 模型加载完成：{model_dir}")
    
    # 5. 生成预测
    top30_features_file = model_dir / 'top30_features.txt'
    with open(top30_features_file, 'r') as f:
        top_30_features = [line.strip() for line in f.readlines()]
    
    X_top30 = X_test[top_30_features]
    predictions_dict = strategy.predict(
        X_top30,
        rr_threshold=2.5,
        prob_threshold=0.75
    )
    
    predictions = pd.DataFrame({
        'predicted_rr': predictions_dict['predicted_rr'],
        'direction': predictions_dict['direction'],
        'holding_period': predictions_dict['holding_period'].clip(1, 30),
        'direction_prob': predictions_dict['direction_prob'],
        'should_trade': predictions_dict['should_trade']
    })
    logger.info(f"✅ 预测完成，信号数：{predictions['should_trade'].sum()}")
    
    # 6. 运行三种策略回测
    logger.info("\n" + "="*60)
    logger.info("开始回测对比")
    logger.info("="*60)
    
    results = []
    
    # 方案A：固定敞口×5
    logger.info("\n【方案 A】固定敞口×5")
    result_a = backtest_with_fixed_exposure(klines_test, predictions, fixed_exposure=5.0)
    results.append(result_a)
    logger.info(f"总收益率：{result_a['total_return']:.2f}%")
    logger.info(f"胜率：{result_a['win_rate']:.2f}%")
    logger.info(f"交易数：{result_a['total_trades']}")
    
    # 方案B：动态敞口（线性规则）
    logger.info("\n【方案 B】动态敞口（线性规则）")
    result_b = backtest_with_dynamic_exposure(klines_test, predictions)
    results.append(result_b)
    logger.info(f"总收益率：{result_b['total_return']:.2f}%")
    logger.info(f"胜率：{result_b['win_rate']:.2f}%")
    logger.info(f"交易数：{result_b['total_trades']}")
    
    # 方案C：凯利公式（平衡模式）
    logger.info("\n【方案 C】凯利公式（平衡模式）")
    result_c = backtest_with_kelly_exposure(
        klines_test, predictions,
        kelly_criterion=0.5,
        risk_mode='balanced'
    )
    results.append(result_c)
    logger.info(f"总收益率：{result_c['total_return']:.2f}%")
    logger.info(f"胜率：{result_c['win_rate']:.2f}%")
    logger.info(f"交易数：{result_c['total_trades']}")
    logger.info(f"最大回撤：{result_c['max_drawdown']:.2f}%")
    logger.info(f"夏普比率：{result_c['sharpe_ratio']:.2f}")
    
    # 方案D：凯利公式（保守模式）
    logger.info("\n【方案 D】凯利公式（保守模式）")
    result_d = backtest_with_kelly_exposure(
        klines_test, predictions,
        kelly_criterion=0.5,
        risk_mode='conservative'
    )
    results.append(result_d)
    logger.info(f"总收益率：{result_d['total_return']:.2f}%")
    logger.info(f"胜率：{result_d['win_rate']:.2f}%")
    logger.info(f"交易数：{result_d['total_trades']}")
    logger.info(f"最大回撤：{result_d['max_drawdown']:.2f}%")
    logger.info(f"夏普比率：{result_d['sharpe_ratio']:.2f}")
    
    # 7. 汇总对比
    logger.info("\n" + "="*60)
    logger.info("📊 策略对比总结")
    logger.info("="*60)
    
    summary_data = []
    for r in results:
        summary_data.append({
            '策略': r['strategy'],
            '总收益 (%)': f"{r['total_return']:.2f}",
            '胜率 (%)': f"{r['win_rate']:.2f}",
            '交易数': r['total_trades'],
            '最大回撤 (%)': f"{r.get('max_drawdown', 0):.2f}" if 'max_drawdown' in r else 'N/A',
            '夏普比率': f"{r.get('sharpe_ratio', 0):.2f}" if 'sharpe_ratio' in r else 'N/A'
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # 8. 找出最优策略
    best_strategy = max(results, key=lambda x: x['total_return'])
    logger.info(f"\n🏆 最优策略：{best_strategy['strategy']}")
    logger.info(f"💰 最高收益率：{best_strategy['total_return']:.2f}%")
    
    # 9. 保存结果
    output_dir = Path('backtest')
    output_dir.mkdir(exist_ok=True)
    
    for r in results:
        trades_df = pd.DataFrame(r['trades'])
        filename = f"kelly_backtest_{r['strategy'].replace('(', '').replace(')', '').replace(' ', '_')}.csv"
        trades_df.to_csv(output_dir / filename, index=False)
    
    logger.info(f"\n✅ 回测结果已保存：{output_dir}")
    
    logger.info("\n" + "="*60)
    logger.info("💡 结论与建议")
    logger.info("="*60)
    
    # 对比分析
    if best_strategy['strategy'].startswith('凯利公式'):
        logger.info("✅ 凯利公式表现最优！")
        logger.info("   建议：可以切换到凯利公式仓位管理")
    else:
        logger.info("⚠️  凯利公式未展现出明显优势")
        logger.info("   可能原因：需要更多历史数据训练")
    
    # 与基准对比
    baseline_return = result_a['total_return']
    kelly_return = result_c['total_return']
    improvement = kelly_return - baseline_return
    
    logger.info(f"\n📈 凯利公式 vs 固定敞口:")
    logger.info(f"   收益提升：{improvement:+.2f}%")
    
    if 'max_drawdown' in result_c and 'max_drawdown' in result_a:
        dd_improvement = result_a.get('max_drawdown', 0) - result_c['max_drawdown']
        logger.info(f"   回撤改善：{dd_improvement:+.2f}%")


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    main()
