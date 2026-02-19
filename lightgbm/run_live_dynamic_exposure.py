#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态敞口策略 - 实盘/模拟盘运行脚本
10倍最大敞口 + 多层风控系统
"""

import time
import json
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


# 状态文件路径（挂载到 Docker 容器外）
STATE_FILE = Path('/app/state/trading_state.json')


def save_trading_state(state_dict, logger):
    """保存交易状态到文件"""
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump(state_dict, f, indent=2)
        logger.debug("💾 状态已保存")
    except Exception as e:
        logger.warning("状态保存失败: %s", e)


def load_trading_state(logger):
    """从文件加载交易状态"""
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            logger.info("📂 已加载交易状态: %s", state)
            return state
    except Exception as e:
        logger.warning("状态加载失败: %s", e)
    return None


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
    max_drawdown_pause = 0.10  # 回撤>10%暂停交易至明日
    rr_threshold = 2.5  # RR阈值（最佳参数，2026-02-17回测验证）
    prob_threshold = 0.75  # 置信度阈值（最佳参数，2026-02-17回测验证）
    
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
    open_entry_idx = 0  # 开仓时的索引（用于计算持仓时间）
    predicted_holding_period = 0  # 预测的持仓周期
    max_profit_pct = 0.0  # 追踪止损：记录最高盈利百分比
    
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
                
                # 尝试恢复持仓状态
                saved_state = load_trading_state(logger)
                if saved_state and saved_state.get('open_position_side') == open_position_side:
                    open_entry_idx = saved_state.get('open_entry_idx', 0)
                    predicted_holding_period = saved_state.get('predicted_holding_period', 0)
                    max_profit_pct = saved_state.get('max_profit_pct', 0.0)
                    open_exposure = saved_state.get('open_exposure', 0.0)
                    logger.info("✅ 已恢复持仓状态: entry_idx=%d, period=%d, max_profit=%.2f%%, exposure=%.2f",
                               open_entry_idx, predicted_holding_period, max_profit_pct*100, open_exposure)
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
                            
                            # 解除每日亏损暂停
                            if trading_paused and pause_reason == 'daily_loss':
                                trading_paused = False
                                pause_reason = None
                                logger.info("✅ 解除每日亏损暂停")
                            
                            # 解除回撤暂停，并重置峰值权益（将回撤归零）
                            elif trading_paused and pause_reason == 'drawdown_pause':
                                trading_paused = False
                                pause_reason = None
                                peak_equity = daily_start_balance
                                logger.info("✅ 解除回撤暂停，回撤已重置为0%%")
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
                    
                    # 回撤检查（只在非暂停状态检查）
                    if peak_equity is not None and not trading_paused:
                        current_drawdown = (peak_equity - current_balance) / peak_equity
                        if current_drawdown > max_drawdown_pause:
                            trading_paused = True
                            pause_reason = 'drawdown_pause'
                            logger.error("🛑 触发回撤暂停 %.2f%%, 暂停交易至明日", current_drawdown * 100)
                    
                    # 更新峰值权益
                    if peak_equity is None or current_balance > peak_equity:
                        peak_equity = current_balance
                except Exception as e:
                    logger.warning("风控检查失败: %s", e)
            
            # 平仓逻辑：持仓周期 + 止损检查
            if enable_trading and open_position_side != "flat":
                try:
                    current_idx = len(klines) - 1
                    bars_held = current_idx - open_entry_idx
                    
                    # 计算当前盈亏
                    if open_position_side == "long":
                        price_change_pct = (current_price - open_entry_price) / open_entry_price
                    else:
                        price_change_pct = (open_entry_price - current_price) / open_entry_price
                    
                    # 止损检查（每次轮询都检查）
                    should_close = False
                    close_reason = ""
                    
                    # 1. 固定止损 -3%
                    if price_change_pct < stop_loss_pct:
                        should_close = True
                        close_reason = f"止损({price_change_pct*100:.2f}% < {stop_loss_pct*100:.1f}%)"
                    
                    # 2. 追踪止损（价格距最高点下降2%）
                    elif price_change_pct > 0.01:  # 盈利>1%启动追踪
                        # 更新最高盈利点
                        if price_change_pct > max_profit_pct:
                            max_profit_pct = price_change_pct
                            logger.info("📈 更新最高盈利: %.2f%%", max_profit_pct * 100)
                            
                            # 更新状态文件
                            save_trading_state({
                                'open_position_side': open_position_side,
                                'open_entry_idx': open_entry_idx,
                                'predicted_holding_period': predicted_holding_period,
                                'max_profit_pct': max_profit_pct,
                                'open_exposure': open_exposure,
                                'open_entry_price': open_entry_price,
                                'open_position_qty': open_position_qty,
                                'timestamp': str(pd.Timestamp.now(tz='UTC'))
                            }, logger)
                        
                        # 价格距最高点下降2%
                        price_drop_from_peak = max_profit_pct - price_change_pct
                        if price_drop_from_peak > 0.02:
                            should_close = True
                            close_reason = f"追踪止损(价格从{max_profit_pct*100:.2f}%回落至{price_change_pct*100:.2f}%, 下跌{price_drop_from_peak*100:.2f}%)"
                    
                    # 3. 持仓周期检查（只在新K线时检查）
                    if not should_close and is_new_bar and bars_held >= predicted_holding_period:
                        should_close = True
                        close_reason = f"持仓周期({bars_held}/{predicted_holding_period})K线"
                    
                    # 执行平仓
                    if should_close:
                        side = "SELL" if open_position_side == "long" else "BUY"
                        position_side = "LONG" if open_position_side == "long" else "SHORT"
                        
                        logger.info("📤 平仓 %s, 数量=%.4f, 原因=%s, 盈亏=%.2f%%",
                                   open_position_side, open_position_qty, close_reason,
                                   price_change_pct * 100)
                        
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
                            open_entry_idx = 0
                            predicted_holding_period = 0
                            max_profit_pct = 0.0  # 重置追踪止损
                            
                            # 删除状态文件
                            try:
                                if STATE_FILE.exists():
                                    STATE_FILE.unlink()
                                    logger.debug("🗑️  状态文件已删除")
                            except Exception as e:
                                logger.warning("删除状态文件失败: %s", e)
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
                            open_entry_idx = len(klines) - 1  # 记录开仓时的索引
                            predicted_holding_period = int(holding_period)  # 记录预测周期
                            max_profit_pct = 0.0  # 初始化追踪止损
                            
                            # 保存状态
                            save_trading_state({
                                'open_position_side': open_position_side,
                                'open_entry_idx': open_entry_idx,
                                'predicted_holding_period': predicted_holding_period,
                                'max_profit_pct': max_profit_pct,
                                'open_exposure': open_exposure,
                                'open_entry_price': open_entry_price,
                                'open_position_qty': open_position_qty,
                                'timestamp': str(pd.Timestamp.now(tz='UTC'))
                            }, logger)
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
