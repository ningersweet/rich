#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态敞口策略 - 实盘/模拟盘运行脚本
10倍最大敞口 + 多层风控系统
"""

import time
import pandas as pd
import numpy as np
import requests
from pathlib import Path

from btc_quant.config import load_config
from btc_quant.data import BINANCE_FAPI_KLINES_ENDPOINT
from btc_quant.execution import BinanceFuturesClient
from btc_quant.features import build_features_and_labels
from btc_quant.monitor import setup_logger
from btc_quant.risk_reward_model import TwoStageRiskRewardStrategy


def fetch_latest_klines(symbol: str, interval: str, limit: int, base_url: str) -> pd.DataFrame:
    """获取最新K线数据"""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(base_url + BINANCE_FAPI_KLINES_ENDPOINT, params=params, timeout=10)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        raise RuntimeError("未获取到最新K线")
    
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
        ],
    )
    df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype("float32")
    return df


def calculate_dynamic_exposure(predicted_rr, direction_prob, current_drawdown=0, 
                               consecutive_losses=0, max_exposure=10.0):
    """
    根据信号质量动态计算最优敞口
    
    参数:
        predicted_rr: 预测盈亏比
        direction_prob: 方向置信度
        current_drawdown: 当前回撤百分比
        consecutive_losses: 连续亏损次数
        max_exposure: 最大敞口限制
    
    返回:
        exposure: 建议敞口（杠杆×仓位），范围 [1.0, max_exposure]
    """
    
    # 基础敞口：基于盈亏比和置信度
    rr_factor = min(predicted_rr / 2.5, 2.0)
    prob_factor = max((direction_prob - 0.5) / 0.5, 0)
    base_exposure = 2.0 + rr_factor * 3.0 + prob_factor * 3.0
    
    # 回撤惩罚
    if current_drawdown > 0.02:
        drawdown_penalty = 1.0 - (current_drawdown - 0.02) * 15
        drawdown_penalty = max(0.3, drawdown_penalty)
    else:
        drawdown_penalty = 1.0
    
    # 连续亏损惩罚
    if consecutive_losses >= 2:
        loss_penalty = 1.0 - min(consecutive_losses - 1, 5) * 0.15
        loss_penalty = max(0.2, loss_penalty)
    else:
        loss_penalty = 1.0
    
    # 最终敞口
    final_exposure = base_exposure * drawdown_penalty * loss_penalty
    final_exposure = np.clip(final_exposure, 1.0, max_exposure)
    
    return final_exposure


def main():
    """主函数"""
    cfg = load_config(Path('config.yaml'))
    logger = setup_logger(cfg)
    client = BinanceFuturesClient(cfg)
    
    # 检查模式
    mode = cfg.api.get("mode", "paper")
    if mode == "paper":
        logger.warning("🔔 当前模式：测试网（paper），API: %s", client.base_url)
        logger.warning("⚠️  将下单到币安测试网！")
    else:
        logger.warning("🔔 当前模式：实盘（live），API: %s", client.base_url)
        logger.warning("⚠️⚠️⚠️  将下单到币安实盘！！！")
    
    enable_trading = True
    symbol = cfg.symbol["name"]
    interval = cfg.symbol["interval"]
    base_url = client.base_url
    
    # 加载盈亏比两阶段模型
    model_dir = Path('models/final_6x_fixed_capital')
    if not model_dir.exists():
        logger.error("模型目录不存在: %s", model_dir)
        return
    
    strategy = TwoStageRiskRewardStrategy()
    strategy.load(model_dir)
    logger.info("✅ 已加载盈亏比两阶段策略模型: %s", model_dir)
    
    # 加载top30特征列表
    top30_features_file = model_dir / 'top30_features.txt'
    if not top30_features_file.exists():
        logger.error("特征文件不存在: %s", top30_features_file)
        return
    
    with open(top30_features_file, 'r') as f:
        top_30_features = [line.strip() for line in f.readlines()]
    logger.info("✅ 已加载TOP30特征，共 %d 个", len(top_30_features))
    
    # 策略参数
    max_exposure = 10.0  # 最大敞口10倍
    stop_loss_pct = -0.03  # 固定止损-3%
    max_daily_loss_pct = -0.20  # 每日最大亏损-20%
    max_drawdown_pause = 0.06  # 回撤>6%暂停交易
    rr_threshold = 2.0  # RR阈值（原2.5，2026-02-13调整为2.0）
    prob_threshold = 0.65  # 置信度阈值（原0.75，2026-02-13调整为0.65）
    
    logger.info("📊 策略参数：")
    logger.info("  最大敞口: %.1f倍", max_exposure)
    logger.info("  固定止损: %.1f%%", stop_loss_pct * 100)
    logger.info("  每日亏损限制: %.1f%%", max_daily_loss_pct * 100)
    logger.info("  回撤暂停阈值: %.1f%%", max_drawdown_pause * 100)
    logger.info("  RR阈值: %.2f", rr_threshold)
    logger.info("  置信度阈值: %.2f", prob_threshold)
    
    poll_interval = int(cfg.live.get("poll_interval_seconds", 60))
    max_new_bars = int(cfg.live.get("max_new_bars", 500))
    
    last_close_time = None
    
    # 持仓状态
    open_position_side = "flat"  # flat / long / short
    open_position_qty = 0.0
    open_entry_price = 0.0
    open_exposure = 0.0  # 当前持仓敞口
    max_profit_pct = 0.0  # 追踪止损用
    
    # 风控状态
    starting_balance = None
    peak_equity = None
    consecutive_losses = 0
    daily_start_balance = None
    current_date = None
    trading_paused = False
    pause_reason = None
    
    if enable_trading:
        try:
            starting_balance = client.get_account_balance_usdt()
            peak_equity = starting_balance
            daily_start_balance = starting_balance
            logger.info("💰 初始余额: %.2f USDT", starting_balance)
            
            # 同步持仓
            pos = client.get_open_position(symbol)
            if pos:
                pos_amt = float(pos.get("positionAmt", 0.0))
                open_entry_price = float(pos.get("entryPrice", 0.0))
                if pos_amt > 0:
                    open_position_side = "long"
                    open_position_qty = pos_amt
                    logger.info("🔄 同步持仓: 做多 %.4f", pos_amt)
                elif pos_amt < 0:
                    open_position_side = "short"
                    open_position_qty = abs(pos_amt)
                    logger.info("🔄 同步持仓: 做空 %.4f", abs(pos_amt))
        except Exception as e:
            logger.exception("获取账户信息失败: %s", e)
            return
    
    logger.info("🚀 开始轮询K线，间隔=%d秒", poll_interval)
    
    while True:
        try:
            # 获取最新K线
            klines = fetch_latest_klines(symbol, interval, max_new_bars, base_url)
            current_close_time = klines.iloc[-1]["close_time"]
            current_price = float(klines.iloc[-1]["close"])
            
            # 检查是否新K线
            is_new_bar = (last_close_time is None or current_close_time > last_close_time)
            
            if is_new_bar:
                logger.info("[新K线] 时间=%s, 价格=%.2f", current_close_time, current_price)
                last_close_time = current_close_time
                
                # 检查日期变化（重置每日亏损）
                current_time = pd.Timestamp.now(tz='UTC')
                if current_date is None or current_time.date() != current_date:
                    current_date = current_time.date()
                    if enable_trading:
                        try:
                            daily_start_balance = client.get_account_balance_usdt()
                            logger.info("📅 新的一天，重置每日起始余额: %.2f USDT", daily_start_balance)
                            if trading_paused and pause_reason == 'daily_loss':
                                trading_paused = False
                                pause_reason = None
                                logger.info("✅ 解除每日亏损暂停")
                        except Exception as e:
                            logger.warning("获取余额失败: %s", e)
            else:
                logger.info("[轮询] 时间=%s, 价格=%.2f (K线内更新)", current_close_time, current_price)
            
            # 每次轮询都预测（无论是否新K线）
            # 构建特征
            feature_label_data = build_features_and_labels(cfg, klines)
            X_full = feature_label_data.features
            
            if len(X_full) == 0:
                logger.warning("特征为空，跳过")
                continue
            
            # 数据对齐
            min_len = min(len(klines), len(X_full))
            klines_aligned = klines.iloc[-min_len:].reset_index(drop=True)
            X_full = X_full.iloc[-min_len:].reset_index(drop=True)
            
            # 提取TOP30特征
            X_top30 = X_full[top_30_features]
            
            # 生成预测
            predictions_dict = strategy.predict(
                X_top30,
                rr_threshold=rr_threshold,
                prob_threshold=prob_threshold
            )
            
            # 取最后一个预测
            should_trade = predictions_dict['should_trade'].iloc[-1]
            predicted_rr = predictions_dict['predicted_rr'].iloc[-1]
            direction = predictions_dict['direction'].iloc[-1]
            direction_prob = predictions_dict['direction_prob'].iloc[-1]
            holding_period = min(predictions_dict['holding_period'].iloc[-1], 30)
            
            logger.info("📊 信号: should_trade=%s, RR=%.2f, direction=%d, prob=%.3f, period=%d",
                       should_trade, predicted_rr, direction, direction_prob, holding_period)
            
            # 风控检查
            if enable_trading and not trading_paused:
                try:
                    current_balance = client.get_account_balance_usdt()
                    
                    # 每日亏损检查
                    if daily_start_balance is not None:
                        daily_loss_pct = (daily_start_balance - current_balance) / daily_start_balance
                        if daily_loss_pct > -max_daily_loss_pct:  # 修复：判断亏损绝对值
                            trading_paused = True
                            pause_reason = 'daily_loss'
                            logger.error("🛑 触发每日最大亏损限制 %.2f%%, 暂停交易", daily_loss_pct * 100)
                    
                    # 回撤检查
                    if peak_equity is not None:
                        current_drawdown = (peak_equity - current_balance) / peak_equity
                        if current_drawdown > max_drawdown_pause:
                            if not trading_paused:
                                trading_paused = True
                                pause_reason = 'drawdown_pause'
                                logger.error("🛑 触发回撤暂停 %.2f%%, 暂停交易", current_drawdown * 100)
                        elif current_balance > peak_equity:
                            peak_equity = current_balance
                            if trading_paused and pause_reason == 'drawdown_pause':
                                # 回撤降低，解除暂停
                                new_drawdown = (peak_equity - current_balance) / peak_equity
                                if new_drawdown < max_drawdown_pause * 0.8:
                                    trading_paused = False
                                    pause_reason = None
                                    logger.info("✅ 回撤降至%.2f%%, 恢复交易", new_drawdown * 100)
                except Exception as e:
                    logger.warning("风控检查失败: %s", e)
            
            # 平仓逻辑：严格止损 + 动态止盈
            if enable_trading and open_position_side != "flat":
                try:
                    # 计算价格变化
                    if open_position_side == "long":
                        price_change_pct = (current_price - open_entry_price) / open_entry_price
                    else:
                        price_change_pct = (open_entry_price - current_price) / open_entry_price
                    
                    # 1. 固定止损检查（-3%，不受任何影响）
                    stop_loss_triggered = (price_change_pct <= stop_loss_pct)
                    
                    # 2. 动态止盈逻辑
                    trailing_stop_triggered = False
                    take_profit_triggered = False
                    
                    if price_change_pct > 0:  # 盈利时才启动
                        max_profit_pct = max(max_profit_pct, price_change_pct)
                        
                        # 2.1 固定止盈：盈利达到3%就锁定部分利润
                        if price_change_pct >= 0.03:
                            take_profit_triggered = True
                        
                        # 2.2 追踪止盈：盈利>1.5%后，回撤40%触发
                        elif price_change_pct > 0.015:
                            profit_retracement = (max_profit_pct - price_change_pct) / max_profit_pct
                            if profit_retracement > 0.40:  # 从高点回落40%
                                trailing_stop_triggered = True
                    
                    # 3. 决定是否平仓（只有止损或止盈才平仓）
                    should_close = stop_loss_triggered or trailing_stop_triggered or take_profit_triggered
                    
                    if should_close:
                        # 平仓
                        side = "SELL" if open_position_side == "long" else "BUY"
                        position_side = "LONG" if open_position_side == "long" else "SHORT"
                        
                        reason = ""
                        if stop_loss_triggered:
                            reason = "🛑 固定止损"
                        elif take_profit_triggered:
                            reason = "🎯 固定止盈(3%)"
                        elif trailing_stop_triggered:
                            reason = "📊 追踪止盈(最高{:.2f}%)".format(max_profit_pct * 100)
                        
                        logger.info("📤 平仓 %s, 数量=%.4f, 原因=%s, 当前盈亏=%.2f%%, 最高盈亏=%.2f%%",
                                   open_position_side, open_position_qty, reason, 
                                   price_change_pct * 100, max_profit_pct * 100)
                        
                        order_res = client.place_market_order(
                            symbol, side, position_side, open_position_qty, reduce_only=True
                        )
                        
                        if order_res.success:
                            logger.info("✅ 平仓成功: %s", order_res.raw)
                            
                            # 更新统计
                            if price_change_pct > 0:
                                consecutive_losses = 0
                            else:
                                consecutive_losses += 1
                            
                            open_position_side = "flat"
                            open_position_qty = 0.0
                            open_entry_price = 0.0
                            open_exposure = 0.0
                            max_profit_pct = 0.0
                        else:
                            logger.error("❌ 平仓失败: %s", order_res.raw)
                
                except Exception as e:
                    logger.exception("平仓逻辑异常: %s", e)
            
            # 开仓逻辑
            if enable_trading and is_new_bar and should_trade and open_position_side == "flat" and not trading_paused:
                try:
                    current_balance = client.get_account_balance_usdt()
                    current_drawdown = (peak_equity - current_balance) / peak_equity if peak_equity else 0
                    
                    # 计算动态敞口
                    optimal_exposure = calculate_dynamic_exposure(
                        predicted_rr=predicted_rr,
                        direction_prob=direction_prob,
                        current_drawdown=current_drawdown,
                        consecutive_losses=consecutive_losses,
                        max_exposure=max_exposure
                    )
                    
                    # 计算开仓数量
                    # 敞口 = 杠杆 × 仓位比例
                    # 这里简化为：名义价值 = 余额 × 敞口
                    notional_value = current_balance * optimal_exposure
                    quantity = notional_value / current_price
                    
                    # 向下取整到合约精度
                    quantity = float(int(quantity * 1000) / 1000)
                    
                    if quantity <= 0:
                        logger.warning("⚠️  开仓数量<=0，跳过")
                    else:
                        desired_side = "long" if direction == 1 else "short"
                        side = "BUY" if desired_side == "long" else "SELL"
                        position_side = "LONG" if desired_side == "long" else "SHORT"
                        
                        logger.info("📥 开仓 %s, 数量=%.4f, 敞口=%.2f倍, RR=%.2f, prob=%.3f",
                                   desired_side, quantity, optimal_exposure, predicted_rr, direction_prob)
                        
                        order_res = client.place_market_order(symbol, side, position_side, quantity)
                        
                        if order_res.success:
                            logger.info("✅ 开仓成功: %s", order_res.raw)
                            open_position_side = desired_side
                            open_position_qty = quantity
                            open_entry_price = current_price
                            open_exposure = optimal_exposure
                            max_profit_pct = 0.0
                        else:
                            logger.error("❌ 开仓失败: %s", order_res.raw)
                
                except Exception as e:
                    logger.exception("开仓逻辑异常: %s", e)
            
            logger.info("💼 持仓=%s, 余额=%.2f, 暂停=%s", open_position_side, 
                       current_balance if enable_trading else 0, trading_paused)
        
        except Exception as e:
            logger.exception("主循环异常: %s", e)
        
        time.sleep(poll_interval)


if __name__ == '__main__':
    main()
