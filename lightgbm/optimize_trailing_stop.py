#!/usr/bin/env python3
"""
移动止盈参数调优脚本
测试不同的trailing_pct参数（30%/40%/50%/60%）
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
from btc_quant.dynamic_stop_loss import TrailingStopManager, calculate_atr_stop_loss


def backtest_with_trailing_stop(klines, predictions, initial_balance=1000.0, max_exposure=10.0,
                                atr_k=3.5, trailing_pct=0.5, min_profit_pct=0.01):
    """
    带移动止盈的回测
    
    参数:
        klines: K线数据
        predictions: 预测结果
        initial_balance: 初始资金
        max_exposure: 最大敞口
        atr_k: ATR止损倍数
        trailing_pct: 利润回撤比例（要优化的参数）
        min_profit_pct: 最小利润要求
    
    返回:
        回测结果字典
    """
    equity = initial_balance
    trades = []
    position = None
    trailing_manager = TrailingStopManager(
        trailing_pct=trailing_pct,
        min_profit_pct=min_profit_pct,
        enable_dynamic_trailing=False
    )
    
    for i in range(len(predictions)):
        current_price = klines.iloc[i]['close']
        current_atr = klines.iloc[i]['atr_14']
        
        # 开仓逻辑
        if position is None and predictions.iloc[i]['should_trade']:
            entry_price = current_price
            predicted_rr = predictions.iloc[i]['predicted_rr']
            direction_prob = predictions.iloc[i]['direction_prob']
            
            # 计算动态敞口
            base_exposure = 1.0
            
            if predicted_rr >= 6.0:
                rr_multiplier = 5.0
            elif predicted_rr >= 4.0:
                rr_multiplier = 3.0 + (predicted_rr - 4.0) * 1.0
            elif predicted_rr >= 2.5:
                rr_multiplier = 1.0 + (predicted_rr - 2.5) * 1.33
            else:
                rr_multiplier = 0.0
            
            if direction_prob >= 0.85:
                prob_multiplier = 5.0
            elif direction_prob >= 0.75:
                prob_multiplier = 3.0 + (direction_prob - 0.75) * 20.0
            elif direction_prob >= 0.65:
                prob_multiplier = 1.0 + (direction_prob - 0.65) * 20.0
            else:
                prob_multiplier = 0.0
            
            optimal_exposure = base_exposure + rr_multiplier + prob_multiplier
            optimal_exposure = min(optimal_exposure, max_exposure)
            
            position_value = equity * 1.0
            
            # 计算ATR止损价
            stop_loss_price, stop_loss_pct = calculate_atr_stop_loss(
                entry_price=entry_price,
                atr=current_atr,
                direction=predictions.iloc[i]['direction'],
                k=atr_k,
                min_stop_loss_pct=0.01,
                max_stop_loss_pct=0.05
            )
            
            position = {
                'side': predictions.iloc[i]['direction'],
                'entry_price': entry_price,
                'entry_idx': i,
                'position_value': position_value,
                'exposure': optimal_exposure,
                'stop_loss_price': stop_loss_price,
                'stop_loss_pct': stop_loss_pct,
                'predicted_rr': predicted_rr,
                'direction_prob': direction_prob,
                'hold_period': int(predictions.iloc[i]['holding_period'])
            }
            
            # 重置移动止盈状态
            trailing_manager.reset()
        
        # 平仓逻辑
        elif position is not None:
            bars_held = i - position['entry_idx']
            should_exit = False
            exit_reason = None
            
            # 1. 检查ATR止损
            if position['side'] == 1:  # 做多
                if current_price <= position['stop_loss_price']:
                    should_exit = True
                    exit_reason = '🛑 ATR止损'
            else:  # 做空
                if current_price >= position['stop_loss_price']:
                    should_exit = True
                    exit_reason = '🛑 ATR止损'
            
            # 2. 更新移动止盈并检查触发
            trailing_manager.update(
                current_price=current_price,
                entry_price=position['entry_price'],
                direction=position['side']
            )
            
            if not should_exit and trailing_manager.is_active:
                if trailing_manager.should_exit(current_price, position['side']):
                    should_exit = True
                    exit_reason = '📈 移动止盈'
            
            # 3. 检查持仓周期到期
            if not should_exit and bars_held >= position['hold_period']:
                should_exit = True
                exit_reason = '⏰ 持仓周期'
            
            # 执行平仓
            if should_exit:
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
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'price_change_pct': price_change_pct * 100,
                    'exposure': position['exposure'],
                    'pnl': pnl,
                    'pnl_pct': price_change_pct * 100 * position['exposure'],
                    'equity_after': equity,
                    'bars_held': bars_held,
                    'reason': exit_reason,
                    'trailing_stop_triggered': (exit_reason == '📈 移动止盈'),
                    'max_profit_reached': trailing_manager.max_profit_reached * 100
                })
                
                position = None
    
    # 最后平仓
    if position is not None:
        final_price = klines.iloc[-1]['close']
        if position['side'] == 1:
            price_change_pct = (final_price - position['entry_price']) / position['entry_price']
        else:
            price_change_pct = (position['entry_price'] - final_price) / position['entry_price']
        
        pnl = initial_balance * price_change_pct * position['exposure']
        equity += pnl
        
        trades.append({
            'entry_time': klines.iloc[position['entry_idx']]['open_time'],
            'exit_time': klines.iloc[-1]['open_time'],
            'side': 'long' if position['side'] == 1 else 'short',
            'entry_price': position['entry_price'],
            'exit_price': final_price,
            'price_change_pct': price_change_pct * 100,
            'exposure': position['exposure'],
            'pnl': pnl,
            'pnl_pct': price_change_pct * 100 * position['exposure'],
            'equity_after': equity,
            'bars_held': len(klines) - position['entry_idx'],
            'reason': '📊 期末平仓',
            'trailing_stop_triggered': False,
            'max_profit_reached': trailing_manager.max_profit_reached * 100
        })
    
    # 计算回测指标
    total_return = (equity - initial_balance) / initial_balance * 100
    win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100 if trades else 0
    atr_stop_trigger_rate = sum(1 for t in trades if t.get('reason') == '🛑 ATR止损') / len(trades) * 100 if trades else 0
    trailing_stop_trigger_rate = sum(1 for t in trades if t.get('trailing_stop_triggered', False)) / len(trades) * 100 if trades else 0
    
    return {
        'initial_balance': initial_balance,
        'final_equity': equity,
        'total_return': total_return,
        'win_rate': win_rate,
        'total_trades': len(trades),
        'trades': trades,
        'atr_stop_trigger_rate': atr_stop_trigger_rate,
        'trailing_stop_trigger_rate': trailing_stop_trigger_rate
    }


def optimize_trailing_pct(klines, predictions, trailing_pcts):
    """
    测试不同移动止盈参数的回测性能
    """
    results = []
    
    for tp in trailing_pcts:
        logger.info(f"\n{'='*60}")
        logger.info(f"测试 trailing_pct={tp:.0%}")
        logger.info(f"{'='*60}")
        
        result = backtest_with_trailing_stop(
            klines,
            predictions,
            initial_balance=1000.0,
            max_exposure=10.0,
            atr_k=3.5,
            trailing_pct=tp
        )
        
        # 计算额外指标
        if result['total_trades'] > 0:
            trades_df = pd.DataFrame(result['trades'])
            
            # 计算平均最大利润
            avg_max_profit = trades_df['max_profit_reached'].mean()
            
            # 计算利润留存率（实际收益/最大可能收益）
            winning_trades = trades_df[trades_df['pnl'] > 0]
            if len(winning_trades) > 0:
                profit_retention = winning_trades['pnl_pct'].mean() / winning_trades['max_profit_reached'].mean() * 100
            else:
                profit_retention = 0
            
            # 计算盈亏比
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] <= 0]
            
            avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['pnl_pct'].mean()) if len(losing_trades) > 0 else 0
            profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            
            # 计算最大回撤
            equity_curve = [1000.0]
            for trade in result['trades']:
                equity_curve.append(trade['equity_after'])
            equity_series = pd.Series(equity_curve)
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max
            max_drawdown = drawdown.min()
            
        else:
            avg_max_profit = 0
            profit_retention = 0
            avg_win = 0
            avg_loss = 0
            profit_loss_ratio = 0
            max_drawdown = 0
        
        results.append({
            'trailing_pct': tp,
            'total_return': result['total_return'],
            'final_equity': result['final_equity'],
            'win_rate': result['win_rate'],
            'total_trades': result['total_trades'],
            'avg_max_profit': avg_max_profit,
            'profit_retention': profit_retention,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'max_drawdown': max_drawdown * 100,
            'atr_stop_trigger_rate': result['atr_stop_trigger_rate'],
            'trailing_stop_trigger_rate': result['trailing_stop_trigger_rate']
        })
        
        logger.info(f"总收益率：{result['total_return']:.2f}%")
        logger.info(f"胜率：{result['win_rate']:.2f}%")
        logger.info(f"平均最大利润：{avg_max_profit:.2f}%")
        logger.info(f"利润留存率：{profit_retention:.1f}%")
        logger.info(f"盈亏比：{profit_loss_ratio:.2f}")
        logger.info(f"最大回撤：{max_drawdown:.2f}%")
        logger.info(f"ATR止损触发率：{result['atr_stop_trigger_rate']:.2f}%")
        logger.info(f"移动止盈触发率：{result['trailing_stop_trigger_rate']:.2f}%")
    
    return pd.DataFrame(results)


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("移动止盈参数trailing_pct优化")
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
    
    # 对齐长度
    min_len = min(len(X_full), len(klines))
    X_full = X_full.iloc[:min_len]
    klines = klines.iloc[:min_len].reset_index(drop=True)
    
    # 添加ATR到klines
    if 'atr_14' in X_full.columns:
        klines['atr_14'] = X_full['atr_14'].values
    else:
        raise ValueError("特征中缺少atr_14列")
    
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
    
    # 6. 测试不同trailing_pct参数
    trailing_pcts = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    logger.info("\n" + "="*60)
    logger.info("开始移动止盈参数优化")
    logger.info("="*60)
    
    results_df = optimize_trailing_pct(klines_test, predictions, trailing_pcts)
    
    # 7. 输出结果表格
    logger.info("\n" + "="*60)
    logger.info("📊 优化结果汇总")
    logger.info("="*60)
    
    # 格式化输出
    display_df = results_df.copy()
    display_df['total_return'] = display_df['total_return'].apply(lambda x: f"{x:.2f}%")
    display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.2f}%")
    display_df['avg_max_profit'] = display_df['avg_max_profit'].apply(lambda x: f"{x:.2f}%")
    display_df['profit_retention'] = display_df['profit_retention'].apply(lambda x: f"{x:.1f}%")
    display_df['profit_loss_ratio'] = display_df['profit_loss_ratio'].apply(lambda x: f"{x:.2f}")
    display_df['max_drawdown'] = display_df['max_drawdown'].apply(lambda x: f"{x:.2f}%")
    display_df['atr_stop_trigger_rate'] = display_df['atr_stop_trigger_rate'].apply(lambda x: f"{x:.2f}%")
    display_df['trailing_stop_trigger_rate'] = display_df['trailing_stop_trigger_rate'].apply(lambda x: f"{x:.2f}%")
    
    print("\n" + display_df.to_string(index=False))
    
    # 8. 找出最优trailing_pct
    best_idx = results_df['total_return'].idxmax()
    best_tp = results_df.loc[best_idx, 'trailing_pct']
    best_return = results_df.loc[best_idx, 'total_return']
    
    logger.info(f"\n🏆 最优trailing_pct: {best_tp:.0%}")
    logger.info(f"💰 最高收益率：{best_return:.2f}%")
    logger.info(f"📈 对应胜率：{results_df.loc[best_idx, 'win_rate']:.2f}%")
    logger.info(f"📉 对应最大回撤：{results_df.loc[best_idx, 'max_drawdown']:.2f}%")
    logger.info(f"🎯 利润留存率：{results_df.loc[best_idx, 'profit_retention']:.1f}%")
    
    # 9. 保存结果
    output_dir = Path('backtest')
    output_dir.mkdir(exist_ok=True)
    
    results_df.to_csv(output_dir / 'trailing_pct_optimization_results.csv', index=False)
    logger.info(f"\n✅ 优化结果已保存：{output_dir / 'trailing_pct_optimization_results.csv'}")
    
    logger.info("\n" + "="*60)
    logger.info("💡 优化建议")
    logger.info("="*60)
    
    if best_tp <= 0.3:
        logger.info("⚠️  最优trailing_pct偏小（≤30%），止盈过紧")
        logger.info("   建议：可能错过趋势行情，考虑使用40%-50%")
    elif best_tp >= 0.6:
        logger.info("⚠️  最优trailing_pct偏大（≥60%），止盈过松")
        logger.info("   建议：利润回撤过多，可能锁定不足")
    else:
        logger.info("✅ 最优trailing_pct适中（40%-50%），平衡了利润保护和趋势跟踪")
    
    # 对比基准（不使用移动止盈）
    from backtest_atr_stop_loss import backtest_with_atr_stop
    baseline_result = backtest_with_atr_stop(
        klines_test, predictions,
        initial_balance=1000.0, max_exposure=10.0,
        enable_trailing_stop=False, atr_k=3.5
    )
    
    logger.info(f"\n📊 对比基准（无移动止盈）:")
    logger.info(f"   基准收益率：{baseline_result['total_return']:.2f}%")
    logger.info(f"   最优trailing_pct提升：{best_return - baseline_result['total_return']:+.2f}%")


if __name__ == '__main__':
    import pandas as pd
    main()
