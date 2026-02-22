#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测引擎模块 - 金字塔加仓版本（使用共享模块）

本模块使用共享的btc_quant.trading模块，确保回测和实盘逻辑一致：
- 使用共享的calculate_dynamic_exposure、should_close_multiple_positions等函数
- 统一仓位管理（Position对象）
- 复用金字塔加仓条件检查
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# 导入共享模块
from btc_quant.trading import (
    Position, TradingState, 
    calculate_dynamic_exposure,
    check_pyramid_conditions,
    calculate_total_pnl_for_positions,
    should_close_multiple_positions,
    position_from_dict, position_to_dict
)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 金字塔加仓参数（从配置文件读取，默认值与实盘保持一致）


def pyramid_backtest_with_compounding(
    klines: pd.DataFrame,
    predictions: pd.DataFrame,
    initial_balance: float = 1000.0,
    max_total_exposure: float = 15.0,
    stop_loss_pct: float = -0.03,
    max_daily_loss_pct: float = -0.20,
    max_drawdown_pause: float = 0.10,
    use_trailing_stop: bool = True,
    pyramid_enabled: bool = True,
    pyramid_profit_threshold: float = 0.01,
    pyramid_min_rr: float = 3.0,
    pyramid_min_prob: float = 0.75,
    pyramid_max_count: int = 3,
    pyramid_min_bars: int = 5
) -> Dict[str, Any]:
    """
    金字塔加仓回测引擎（使用共享模块）
    
    参数:
        klines: K线数据DataFrame
        predictions: 预测数据DataFrame
        initial_balance: 初始资金
        max_total_exposure: 最大总敞口限制
        stop_loss_pct: 固定止损百分比
        max_daily_loss_pct: 每日最大亏损百分比
        max_drawdown_pause: 回撤暂停阈值
        use_trailing_stop: 是否使用追踪止损
        pyramid_enabled: 是否启用金字塔加仓
        pyramid_profit_threshold: 盈利阈值（>此值后允许加仓）
        pyramid_min_rr: 加仓信号最小盈亏比阈值
        pyramid_min_prob: 加仓信号最小概率阈值
        pyramid_max_count: 最大加仓次数
        pyramid_min_bars: 距上次加仓最小K线数
        
    返回:
        Dict包含回测结果
    """
    # 初始化交易状态
    trading_state = TradingState(
        equity=initial_balance,
        peak_equity=initial_balance,
        daily_start_equity=initial_balance,
        consecutive_losses=0,
        daily_loss_paused=False,
        drawdown_paused=False
    )
    
    # 仓位管理
    positions: List[Position] = []  # 多仓位列表（使用Position对象）
    trades = []
    current_date = None
    last_pyramid_time: Optional[pd.Timestamp] = None
    
    for i in range(len(predictions)):
        current_time = klines.iloc[i]['open_time']
        current_price = klines.iloc[i]['close']
        current_day = pd.Timestamp(current_time).date()
        
        # 每日重置逻辑（与实盘一致）
        if current_date != current_day:
            current_date = current_day
            trading_state.daily_start_equity = trading_state.equity
            if trading_state.daily_loss_paused:
                trading_state.daily_loss_paused = False
                logger.info(f"[{current_time}] 新的一天,恢复交易(每日亏损暂停已解除)")
            if trading_state.drawdown_paused:
                trading_state.drawdown_paused = False
                logger.info(f"[{current_time}] 新的一天,恢复交易(回撤暂停已解除)")
        
        # 计算当前回撤
        if trading_state.peak_equity > 0:
            current_drawdown = (trading_state.peak_equity - trading_state.equity) / trading_state.peak_equity
        else:
            current_drawdown = 0
        
        # 持仓管理 - 使用共享模块的平仓逻辑
        if len(positions) > 0:
            # 计算已持仓周期（以首仓为准）
            first_entry_time = positions[0].entry_time
            if first_entry_time is not None:
                time_diff = (current_time - first_entry_time).total_seconds()
                current_hold_period = int(time_diff / (15 * 60))  # 15分钟K线
            else:
                current_hold_period = 0
            
            # 使用共享模块判断是否应该平仓
            should_close, close_reason, stop_loss_hit, trailing_stop_hit = should_close_multiple_positions(
                positions=positions,
                current_price=current_price,
                stop_loss_pct=stop_loss_pct,
                use_trailing_stop=use_trailing_stop,
                trailing_stop_trigger=0.01,
                trailing_stop_distance=0.02,
                current_hold_period=current_hold_period,
                kline_interval_minutes=15
            )
            
            # 平仓处理
            if should_close:
                # 计算总盈亏百分比和盈亏金额
                total_pnl_pct, total_pnl, avg_entry_price = calculate_total_pnl_for_positions(
                    positions, current_price, trading_state.equity
                )
                
                # 更新权益
                trading_state.equity += total_pnl
                
                if trading_state.equity > trading_state.peak_equity:
                    trading_state.peak_equity = trading_state.equity
                
                # 更新连续亏损次数
                if total_pnl <= 0:
                    trading_state.consecutive_losses += 1
                else:
                    trading_state.consecutive_losses = 0
                
                # 计算总敞口
                total_exposure = sum(p.exposure for p in positions)
                
                # 记录交易（合并所有仓位）
                trades.append({
                    'entry_time': positions[0].entry_time,
                    'exit_time': current_time,
                    'side': positions[0].side,
                    'entry_price': avg_entry_price,
                    'exit_price': current_price,
                    'exposure': total_exposure,
                    'pyramid_count': len(positions),
                    'price_change_pct': ((current_price - avg_entry_price) / avg_entry_price * 100 if positions[0].side == 'long' else (avg_entry_price - current_price) / avg_entry_price * 100),
                    'pnl_pct': total_pnl_pct * 100,
                    'pnl': total_pnl,
                    'equity_after': trading_state.equity,
                    'close_reason': close_reason,
                    'stop_loss_hit': stop_loss_hit,
                    'trailing_stop_hit': trailing_stop_hit,
                    'consecutive_losses': trading_state.consecutive_losses
                })
                
                # 清空仓位
                positions.clear()
                last_pyramid_time = None
                
                # 风控检查（与实盘一致）
                daily_loss_pct = (trading_state.equity - trading_state.daily_start_equity) / trading_state.daily_start_equity
                if daily_loss_pct < max_daily_loss_pct:
                    trading_state.daily_loss_paused = True
                    logger.warning(f"[{current_time}] 触发每日最大亏损限制: {daily_loss_pct*100:.2f}%, 暂停交易至明日")
                
                current_drawdown = (trading_state.peak_equity - trading_state.equity) / trading_state.peak_equity if trading_state.peak_equity > 0 else 0
                if current_drawdown > max_drawdown_pause:
                    trading_state.drawdown_paused = True
                    logger.warning(f"[{current_time}] 触发回撤暂停: {current_drawdown*100:.2f}%, 暂停交易至明日")
        
        # 开仓/加仓逻辑
        if predictions.iloc[i]['should_trade']:
            if trading_state.daily_loss_paused or trading_state.drawdown_paused:
                continue
            
            # 使用共享模块计算动态敞口
            new_exposure = calculate_dynamic_exposure(
                predicted_rr=predictions.iloc[i]['predicted_rr'],
                direction_prob=predictions.iloc[i]['direction_prob'],
                current_drawdown=current_drawdown,
                consecutive_losses=trading_state.consecutive_losses,
                max_exposure=10.0
            )
            
            # 情况1：无持仓，开新仓
            if len(positions) == 0:
                new_position = Position(
                    side='long' if predictions.iloc[i]['direction'] == 1 else 'short',
                    entry_price=current_price,
                    entry_time=current_time,
                    exposure=new_exposure,
                    hold_period=int(predictions.iloc[i]['holding_period']),
                    quantity=new_exposure * trading_state.equity / current_price,  # 计算数量
                    peak_pnl_pct=0.0,
                    peak_price=current_price
                )
                positions.append(new_position)
                last_pyramid_time = current_time
            
            # 情况2：有持仓，检查加仓
            elif pyramid_enabled and len(positions) < pyramid_max_count:
                # 使用共享模块检查加仓条件
                can_pyramid = check_pyramid_conditions(
                    positions=positions,
                    current_price=current_price,
                    new_signal_direction=predictions.iloc[i]['direction'],
                    new_signal_rr=predictions.iloc[i]['predicted_rr'],
                    new_signal_prob=predictions.iloc[i]['direction_prob'],
                    last_pyramid_time=last_pyramid_time,
                    current_time=current_time,
                    pyramid_profit_threshold=pyramid_profit_threshold,
                    pyramid_min_rr=pyramid_min_rr,
                    pyramid_min_prob=pyramid_min_prob,
                    pyramid_min_bars=pyramid_min_bars,
                    max_total_exposure=max_total_exposure,
                    new_exposure=new_exposure,
                    kline_interval_minutes=15
                )
                
                if can_pyramid:
                    # 计算加仓数量
                    notional_value = trading_state.equity * new_exposure
                    quantity = notional_value / current_price
                    quantity = float(int(quantity * 1000) / 1000)  # 精度控制
                    
                    if quantity > 0:
                        new_position = Position(
                            side='long' if predictions.iloc[i]['direction'] == 1 else 'short',
                            entry_price=current_price,
                            entry_time=current_time,
                            exposure=new_exposure,
                            hold_period=positions[0].hold_period,  # 继承首仓周期
                            quantity=quantity,
                            peak_pnl_pct=0.0,
                            peak_price=current_price
                        )
                        positions.append(new_position)
                        last_pyramid_time = current_time
                        
                        # 计算当前总敞口
                        total_exposure = sum(p.exposure for p in positions)
                        logger.info(f"[{current_time}] 加仓成功! 第{len(positions)}仓, 敞口{new_exposure:.1f}倍, 总敞口{total_exposure:.1f}倍")
    
    # 强制平仓（回测结束）
    if len(positions) > 0:
        current_price = klines.iloc[-1]['close']
        total_pnl_pct, total_pnl, avg_entry_price = calculate_total_pnl_for_positions(
            positions, current_price, trading_state.equity
        )
        trading_state.equity += total_pnl
        
        total_exposure = sum(p.exposure for p in positions)
        
        trades.append({
            'entry_time': positions[0].entry_time,
            'exit_time': klines.iloc[-1]['open_time'],
            'side': positions[0].side,
            'entry_price': avg_entry_price,
            'exit_price': current_price,
            'exposure': total_exposure,
            'pyramid_count': len(positions),
            'price_change_pct': ((current_price - avg_entry_price) / avg_entry_price * 100 if positions[0].side == 'long' else (avg_entry_price - current_price) / avg_entry_price * 100),
            'pnl_pct': total_pnl_pct * 100,
            'pnl': total_pnl,
            'equity_after': trading_state.equity,
            'close_reason': '强制平仓',
            'stop_loss_hit': False,
            'trailing_stop_hit': False,
            'consecutive_losses': trading_state.consecutive_losses
        })
    
    # 统计计算
    total_return = (trading_state.equity / initial_balance - 1) * 100
    
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        winning_trades = (trades_df['pnl'] > 0).sum()
        win_rate = winning_trades / len(trades) * 100
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].mean()) if (trades_df['pnl'] <= 0).sum() > 0 else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # 计算最大回撤
        peak = trades_df['equity_after'].expanding().max()
        max_drawdown = (peak - trades_df['equity_after']) / peak * 100
        max_drawdown = max_drawdown.max()
        
        avg_exposure = trades_df['exposure'].mean()
        pyramid_trades = (trades_df['pyramid_count'] > 1).sum()
        avg_pyramid_exposure = trades_df[trades_df['pyramid_count'] > 1]['exposure'].mean() if pyramid_trades > 0 else 0
    else:
        win_rate, profit_loss_ratio, max_drawdown, avg_exposure, pyramid_trades, avg_pyramid_exposure, trades_df = 0, 0, 0, 0, 0, 0, None
    
    return {
        'total_return': total_return,
        'final_equity': trading_state.equity,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'max_drawdown': max_drawdown,
        'avg_exposure': avg_exposure,
        'pyramid_trades': pyramid_trades,
        'avg_pyramid_exposure': avg_pyramid_exposure,
        'trades': trades_df
    }


def run_backtest() -> None:
    """主回测流程"""
    logger.info("=" * 80)
    logger.info("🔍 开始金字塔加仓回测（使用共享模块）")
    logger.info("=" * 80)
    
    try:
        from btc_quant.config import load_config
        from btc_quant.data import load_klines
        from btc_quant.features import build_features_and_labels
        from btc_quant.risk_reward_model import TwoStageRiskRewardStrategy
        
        # 回测配置（与backtest_engine.py保持一致）
        BACKTEST_START = '2026-01-01T00:00:00Z'
        BACKTEST_END = '2026-02-07T23:59:59Z'
        MODEL_DIR = Path('models/final_2025_dynamic')
        INITIAL_BALANCE = 1000.0
        MAX_EXPOSURE = 10.0
        STOP_LOSS_PCT = -0.03
        MAX_DAILY_LOSS_PCT = -0.20
        MAX_DRAWDOWN_PAUSE = 0.10
        USE_TRAILING_STOP = True
        MAX_HOLDING_PERIOD = 20
        RR_THRESHOLD = 1.0
        PROB_THRESHOLD = 0.0
        
        # 加载配置
        cfg = load_config()
        
        # 从配置文件读取金字塔参数
        enhanced_cfg = cfg.raw.get('enhanced', {})
        pyramid_enabled = enhanced_cfg.get('enable_pyramid', True)
        pyramid_profit_threshold = enhanced_cfg.get('pyramid_profit_threshold', 0.01)
        pyramid_min_rr = enhanced_cfg.get('pyramid_min_rr', 3.0)
        pyramid_min_prob = enhanced_cfg.get('pyramid_min_prob', 0.75)
        pyramid_max_count = enhanced_cfg.get('pyramid_max_count', 3)
        pyramid_min_bars = enhanced_cfg.get('pyramid_min_bars', 5)
        max_total_exposure = enhanced_cfg.get('max_total_exposure', 15.0)
        
        logger.info(f"\n回测时间范围:{BACKTEST_START}至{BACKTEST_END}")
        logger.info("\n加载K线数据...")
        klines_all = load_klines(cfg)
        klines_all['close_time'] = pd.to_datetime(klines_all['close_time'])
        backtest_start_ts, backtest_end_ts = pd.Timestamp(BACKTEST_START), pd.Timestamp(BACKTEST_END)
        klines_backtest = klines_all[(klines_all['close_time'] >= backtest_start_ts) & (klines_all['close_time'] <= backtest_end_ts)].reset_index(drop=True)
        logger.info(f"回测集K线数量:{len(klines_backtest)}")
        logger.info(f"回测集时间范围:{klines_backtest['close_time'].min()}至{klines_backtest['close_time'].max()}")
        
        logger.info("\n构建特征...")
        feature_label_data = build_features_and_labels(cfg, klines_backtest)
        X_backtest_full = feature_label_data.features.reset_index(drop=True)
        min_len = min(len(X_backtest_full), len(klines_backtest))
        X_backtest_full = X_backtest_full.iloc[:min_len]
        klines_backtest = klines_backtest.iloc[:min_len].reset_index(drop=True)
        logger.info(f"对齐后样本数:{len(X_backtest_full)}")
        
        logger.info(f"\n加载模型:{MODEL_DIR}")
        strategy = TwoStageRiskRewardStrategy()
        strategy.load(MODEL_DIR)
        with open(MODEL_DIR / 'top30_features.txt', 'r') as f:
            top_30_features = [line.strip() for line in f.readlines()]
        logger.info(f"特征数量:{len(top_30_features)}")
        X_backtest_top30 = X_backtest_full[top_30_features]
        
        logger.info("\n生成预测信号...")
        predictions_dict = strategy.predict(X_backtest_top30, rr_threshold=RR_THRESHOLD, prob_threshold=PROB_THRESHOLD)
        predictions = pd.DataFrame({
            'predicted_rr': predictions_dict['predicted_rr'],
            'direction': predictions_dict['direction'],
            'holding_period': predictions_dict['holding_period'].clip(1, MAX_HOLDING_PERIOD),
            'direction_prob': predictions_dict['direction_prob'],
            'should_trade': predictions_dict['should_trade']
        })
        logger.info(f"总样本数:{len(predictions)}")
        logger.info(f"应交易样本:{predictions['should_trade'].sum()}")
        logger.info(f"交易比例:{predictions['should_trade'].sum()/len(predictions)*100:.2f}%")
        
        logger.info(f"📊 数据准备完成: K线{len(klines_backtest)}条, 预测{len(predictions)}条")
        
        # 显示金字塔参数
        logger.info("🏗️ 金字塔加仓参数:")
        logger.info("  启用状态: %s", "启用" if pyramid_enabled else "禁用")
        if pyramid_enabled:
            logger.info("  加仓条件: 盈利>%.1f%%, RR≥%.1f, 概率≥%.2f", 
                       pyramid_profit_threshold * 100, pyramid_min_rr, pyramid_min_prob)
            logger.info("  最多加仓次数: %d, 最小K线间隔: %d", pyramid_max_count, pyramid_min_bars)
            logger.info("  总敞口上限: %.1f倍", max_total_exposure)
        
        # 运行回测
        result = pyramid_backtest_with_compounding(
            klines=klines_backtest,
            predictions=predictions,
            initial_balance=INITIAL_BALANCE,
            max_total_exposure=max_total_exposure,
            stop_loss_pct=STOP_LOSS_PCT,
            max_daily_loss_pct=MAX_DAILY_LOSS_PCT,
            max_drawdown_pause=MAX_DRAWDOWN_PAUSE,
            use_trailing_stop=USE_TRAILING_STOP,
            pyramid_enabled=pyramid_enabled,
            pyramid_profit_threshold=pyramid_profit_threshold,
            pyramid_min_rr=pyramid_min_rr,
            pyramid_min_prob=pyramid_min_prob,
            pyramid_max_count=pyramid_max_count,
            pyramid_min_bars=pyramid_min_bars
        )
        
        # 输出结果
        logger.info("📈 回测完成!")
        logger.info(f"  总收益率: {result['total_return']:.2f}%")
        logger.info(f"  最终权益: {result['final_equity']:.2f} USDT")
        logger.info(f"  交易次数: {result['total_trades']}")
        logger.info(f"  胜率: {result['win_rate']:.2f}%")
        logger.info(f"  盈亏比: {result['profit_loss_ratio']:.2f}")
        logger.info(f"  最大回撤: {result['max_drawdown']:.2f}%")
        logger.info(f"  平均敞口: {result['avg_exposure']:.2f}倍")
        logger.info(f"  包含加仓的交易数: {result['pyramid_trades']}")
        logger.info(f"  加仓交易平均敞口: {result['avg_pyramid_exposure']:.2f}倍")
        
        return result
        
    except Exception as e:
        logger.exception(f"回测异常: {e}")
        return None


if __name__ == "__main__":
    run_backtest()