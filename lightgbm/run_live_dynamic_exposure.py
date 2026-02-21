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
from btc_quant.email_notifier import EmailNotifier
from btc_quant.trading import (
    Position, TradingState, calculate_dynamic_exposure,
    should_open_position, should_close_position, calculate_pnl,
    update_trading_state, reset_daily_state,
    position_from_dict, position_to_dict
)


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
            
            # 版本兼容性处理
            version = state.get('version', 1)
            if version == 1:
                # 旧版本：单个仓位，转换为positions列表
                if state.get('open_position_side', 'flat') != 'flat':
                    # 创建单个仓位对象
                    pos = Position(
                        side=state['open_position_side'],
                        entry_price=state.get('open_entry_price', 0.0),
                        entry_time=pd.Timestamp(state.get('open_entry_time')).to_pydatetime() if state.get('open_entry_time') else None,
                        exposure=state.get('open_exposure', 0.0),
                        hold_period=state.get('predicted_holding_period', 0),
                        quantity=state.get('open_position_qty', 0.0),
                        peak_pnl_pct=state.get('max_profit_pct', 0.0),
                        peak_price=0.0
                    )
                    positions = [pos]
                    last_pyramid_time = None
                else:
                    positions = []
                    last_pyramid_time = None
                # 更新state字典以包含新字段
                state['version'] = 2
                state['positions'] = [position_to_dict(p) for p in positions]
                state['last_pyramid_time'] = None
            else:
                # 版本2：直接使用positions字段
                positions = [position_from_dict(d) for d in state.get('positions', [])]
                last_pyramid_time = pd.Timestamp(state['last_pyramid_time']).to_pydatetime() if state.get('last_pyramid_time') else None
                state['positions'] = [position_to_dict(p) for p in positions]  # 确保序列化格式一致
                state['last_pyramid_time'] = str(last_pyramid_time) if last_pyramid_time else None
            
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
    model_dir = Path('models/final_2024_dynamic')
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
    
    # 金字塔加仓参数（从配置文件读取）
    enhanced_cfg = cfg.raw.get('enhanced', {})
    pyramid_enabled = enhanced_cfg.get('enable_pyramid', True)  # 启用金字塔加仓
    pyramid_profit_threshold = enhanced_cfg.get('pyramid_profit_threshold', 0.01)  # 盈利>1%后允许加仓
    pyramid_min_rr = enhanced_cfg.get('pyramid_min_rr', 3.0)  # 加仓信号盈亏比阈值
    pyramid_min_prob = enhanced_cfg.get('pyramid_min_prob', 0.75)  # 加仓信号概率阈值
    pyramid_max_count = enhanced_cfg.get('pyramid_max_count', 3)  # 最多加仓次数
    pyramid_min_bars = enhanced_cfg.get('pyramid_min_bars', 5)  # 距上次加仓最小K线数
    max_total_exposure = enhanced_cfg.get('max_total_exposure', 15.0)  # 总敞口上限（含加仓）
    
    logger.info("📊 策略参数：")
    logger.info("  最大敞口: %.1f倍", max_exposure)
    logger.info("  固定止损: %.1f%%", stop_loss_pct * 100)
    logger.info("  每日亏损限制: %.1f%%", max_daily_loss_pct * 100)
    logger.info("  回撤暂停阈值: %.1f%%", max_drawdown_pause * 100)
    logger.info("  RR阈值: %.2f", rr_threshold)
    logger.info("  置信度阈值: %.2f", prob_threshold)
    logger.info("  金字塔加仓: %s", "启用" if pyramid_enabled else "禁用")
    if pyramid_enabled:
        logger.info("    加仓条件: 盈利>%.1f%%, RR≥%.1f, 概率≥%.2f", pyramid_profit_threshold * 100, pyramid_min_rr, pyramid_min_prob)
        logger.info("    最多加仓次数: %d, 最小K线间隔: %d", pyramid_max_count, pyramid_min_bars)
        logger.info("    总敞口上限: %.1f倍", max_total_exposure)
    
    # 初始化邮件通知器
    email_notifier = None
    try:
        email_cfg = cfg.raw.get('email', {})
        if email_cfg.get('enabled', False):
            email_notifier = EmailNotifier(
                smtp_host=email_cfg['smtp_host'],
                smtp_port=email_cfg['smtp_port'],
                sender_email=email_cfg['sender_email'],
                sender_password=email_cfg['sender_password'],
                receiver_email=email_cfg['receiver_email'],
                enabled=True
            )
        else:
            logger.info("✉️  邮件通知未启用")
    except Exception as e:
        logger.warning("邮件通知初始化失败: %s", e)
    
    poll_interval = int(cfg.live.get("poll_interval_seconds", 60))
    max_new_bars = int(cfg.live.get("max_new_bars", 500))
    
    last_close_time = None
    
    # 持仓状态（支持金字塔加仓）
    positions = []  # Position对象列表，支持多个仓位
    last_pyramid_time = None  # 上次加仓时间
    open_position_side = "flat"  # flat / long / short（根据positions推导）
    open_position_qty = 0.0  # 总数量（根据positions计算）
    open_entry_price = 0.0  # 平均入场价（根据positions计算）
    open_exposure = 0.0  # 总敞口（根据positions计算）
    open_entry_time = None  # 首仓开仓时间（根据positions推导）
    predicted_holding_period = 0  # 预测的持仓周期（首仓周期）
    max_profit_pct = 0.0  # 追踪止损：记录最高盈利百分比（所有仓位中最高）
    
    def update_derived_position_vars():
        """根据positions列表更新所有派生变量"""
        nonlocal open_position_side, open_position_qty, open_entry_price, open_exposure
        nonlocal open_entry_time, predicted_holding_period, max_profit_pct, position
        
        if not positions:
            open_position_side = "flat"
            open_position_qty = 0.0
            open_entry_price = 0.0
            open_exposure = 0.0
            open_entry_time = None
            predicted_holding_period = 0
            max_profit_pct = 0.0
            position = None
            return
        
        # 首仓信息
        first_pos = positions[0]
        open_position_side = first_pos.side
        open_entry_time = first_pos.entry_time
        predicted_holding_period = first_pos.hold_period
        
        # 计算总量、总敞口、加权平均入场价
        total_qty = sum(p.quantity for p in positions)
        total_exposure = sum(p.exposure for p in positions)
        if total_qty > 0:
            avg_entry_price = sum(p.entry_price * p.quantity for p in positions) / total_qty
        else:
            avg_entry_price = first_pos.entry_price
        
        open_position_qty = total_qty
        open_exposure = total_exposure
        open_entry_price = avg_entry_price
        
        # 计算所有仓位中的最高盈利百分比
        max_profit_pct = max((p.peak_pnl_pct for p in positions), default=0.0)
        
        # 兼容性变量：position指向首仓（用于共享模块）
        position = first_pos
    
    # 统一持仓对象（使用共享模块，兼容性变量）
    position = None  # Position对象（兼容性，从positions[0]派生）
    
    # 风控状态
    starting_balance = None
    peak_equity = None
    consecutive_losses = 0
    daily_start_balance = None
    current_date = None
    daily_loss_paused = False
    drawdown_paused = False
    
    # 统一交易状态（使用共享模块）
    trading_state = TradingState(
        equity=0.0,
        peak_equity=0.0,
        daily_start_equity=0.0,
        consecutive_losses=consecutive_losses,
        daily_loss_paused=daily_loss_paused,
        drawdown_paused=drawdown_paused
    )
    
    if enable_trading:
        try:
            starting_balance = client.get_account_balance_usdt()
            peak_equity = starting_balance
            daily_start_balance = starting_balance
            trading_state.equity = starting_balance
            trading_state.peak_equity = starting_balance
            trading_state.daily_start_equity = starting_balance
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
                
                # 尝试恢复持仓状态（支持金字塔加仓）
                saved_state = load_trading_state(logger)
                if saved_state and saved_state.get('positions'):
                    positions = [position_from_dict(d) for d in saved_state.get('positions', [])]
                    last_pyramid_time_str = saved_state.get('last_pyramid_time')
                    last_pyramid_time = pd.to_datetime(last_pyramid_time_str) if last_pyramid_time_str else None
                    # 更新派生变量
                    update_derived_position_vars()
                    logger.info("✅ 已恢复持仓状态: 仓位数量=%d, 总敞口=%.2f, 最高盈利=%.2f%%",
                               len(positions), open_exposure, max_profit_pct*100)
                elif saved_state and saved_state.get('open_position_side') != 'flat':
                    # 旧版本状态恢复（兼容性）
                    open_entry_time = saved_state.get('open_entry_time')
                    if open_entry_time:
                        open_entry_time = pd.to_datetime(open_entry_time)
                    predicted_holding_period = saved_state.get('predicted_holding_period', 0)
                    max_profit_pct = saved_state.get('max_profit_pct', 0.0)
                    open_exposure = saved_state.get('open_exposure', 0.0)
                    logger.info("✅ 已恢复持仓状态: entry_time=%s, period=%d, max_profit=%.2f%%, exposure=%.2f",
                               open_entry_time, predicted_holding_period, max_profit_pct*100, open_exposure)
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
                            
                            # 使用共享模块重置每日状态
                            reset_daily_state(trading_state, daily_start_balance)
                            # 如果之前回撤暂停，重置峰值权益
                            if drawdown_paused:
                                trading_state.peak_equity = daily_start_balance
                            
                            # 同步旧变量（用于兼容性）
                            daily_loss_paused = trading_state.daily_loss_paused
                            drawdown_paused = trading_state.drawdown_paused
                            peak_equity = trading_state.peak_equity
                            daily_start_balance = trading_state.daily_start_equity
                            logger.info("✅ 已重置每日状态，解除所有暂停")
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
            if enable_trading and not daily_loss_paused and not drawdown_paused:
                try:
                    current_balance = client.get_account_balance_usdt()
                    
                    # 更新统一交易状态
                    trading_state.equity = current_balance
                    if current_balance > trading_state.peak_equity:
                        trading_state.peak_equity = current_balance
                    
                    # 使用共享模块更新风控状态
                    update_trading_state(
                        trading_state=trading_state,
                        pnl=0.0,  # 无新交易，仅检查风控
                        current_time=pd.Timestamp.now(tz='UTC'),
                        max_daily_loss_pct=max_daily_loss_pct,
                        max_drawdown_pause=max_drawdown_pause
                    )
                    
                    # 同步旧变量（用于兼容性）
                    daily_loss_paused = trading_state.daily_loss_paused
                    drawdown_paused = trading_state.drawdown_paused
                    peak_equity = trading_state.peak_equity
                    daily_start_balance = trading_state.daily_start_equity
                    consecutive_losses = trading_state.consecutive_losses
                    
                    # 检查是否触发新的风控暂停，发送邮件通知
                    if daily_loss_paused:
                        daily_loss_pct = (trading_state.daily_start_equity - trading_state.equity) / trading_state.daily_start_equity
                        logger.error("🛑 触发每日最大亏损限制 %.2f%%, 暂停交易至明日", daily_loss_pct * 100)
                        if email_notifier:
                            try:
                                email_notifier.notify_risk_alert(
                                    alert_type="每日亏损限制",
                                    message=f"每日亏损达到 {daily_loss_pct * 100:.2f}%，已暂停交易",
                                    balance=current_balance
                                )
                            except Exception as e:
                                logger.warning("风控邮件通知失败: %s", e)
                    
                    if drawdown_paused:
                        current_drawdown = (trading_state.peak_equity - trading_state.equity) / trading_state.peak_equity
                        logger.error("🛑 触发回撤暂停 %.2f%%, 暂停交易至明日", current_drawdown * 100)
                        if email_notifier:
                            try:
                                email_notifier.notify_risk_alert(
                                    alert_type="回撤暂停",
                                    message=f"回撤达到 {current_drawdown * 100:.2f}%，已暂停交易至明日",
                                    current_drawdown=current_drawdown * 100,
                                    balance=current_balance
                                )
                            except Exception as e:
                                logger.warning("风控邮件通知失败: %s", e)
                                
                except Exception as e:
                    logger.warning("风控检查失败: %s", e)
            
            # 平仓逻辑：持仓周期 + 止损检查（支持金字塔加仓）
            if enable_trading and len(positions) > 0:
                try:
                    # 计算持仓K线数（基于首仓时间）
                    bars_held = 0
                    if positions[0].entry_time is not None:
                        time_diff = (current_close_time - positions[0].entry_time).total_seconds()
                        bars_held = int(time_diff / (15 * 60))  # 15分钟K线
                    
                    # 计算多仓位总盈亏百分比和总敞口
                    total_pnl_pct = 0.0
                    total_exposure = 0.0
                    total_qty = 0.0
                    weighted_entry_sum = 0.0
                    peak_pnl_pct = 0.0
                    peak_updated = False
                    
                    for pos in positions:
                        # 计算单个仓位价格变化百分比
                        if pos.side == 'long':
                            price_change_pct = (current_price - pos.entry_price) / pos.entry_price
                        else:
                            price_change_pct = (pos.entry_price - current_price) / pos.entry_price
                        
                        # 单个仓位盈亏百分比
                        pnl_pct = price_change_pct * pos.exposure
                        total_pnl_pct += pnl_pct
                        total_exposure += pos.exposure
                        total_qty += pos.quantity
                        weighted_entry_sum += pos.entry_price * pos.quantity
                        
                        # 更新单个仓位的峰值（用于追踪止损）
                        if pnl_pct > pos.peak_pnl_pct:
                            pos.peak_pnl_pct = pnl_pct
                            pos.peak_price = current_price
                            peak_updated = True
                        
                        # 更新全局峰值
                        if pnl_pct > peak_pnl_pct:
                            peak_pnl_pct = pnl_pct
                    
                    # 计算平均入场价（用于日志和邮件）
                    avg_entry_price = weighted_entry_sum / total_qty if total_qty > 0 else positions[0].entry_price
                    
                    # 统一平仓条件检查（类似金字塔回测逻辑）
                    should_close = False
                    close_reason = ""
                    stop_loss_hit = False
                    trailing_stop_hit = False
                    
                    # 1. 固定止损
                    if total_pnl_pct <= stop_loss_pct:
                        should_close = True
                        close_reason = f"固定止损({total_pnl_pct*100:.2f}% ≤ {stop_loss_pct*100:.1f}%)"
                        stop_loss_hit = True
                    
                    # 2. 追踪止损（任一仓位盈利>1%后启用）
                    elif peak_pnl_pct > 0.01:  # trailing_stop_trigger
                        # 计算从最高点的回撤
                        pnl_drop_from_peak = peak_pnl_pct - total_pnl_pct
                        if pnl_drop_from_peak > 0.02:  # trailing_stop_distance
                            should_close = True
                            close_reason = f"追踪止损(从{peak_pnl_pct*100:.2f}%回落{total_pnl_pct*100:.2f}%)"
                            trailing_stop_hit = True
                    
                    # 3. 持仓周期（以首仓为准）
                    elif bars_held >= positions[0].hold_period:
                        should_close = True
                        close_reason = f"持仓周期({bars_held}/{positions[0].hold_period})"
                    
                    # 更新派生变量（当前盈亏百分比和价格变化百分比用于日志）
                    # 注意：这里我们使用总盈亏百分比和平均入场价来计算价格变化百分比
                    # 对于日志和邮件，我们使用总盈亏百分比和平均入场价
                    current_pnl_pct = total_pnl_pct
                    if positions[0].side == 'long':
                        price_change_pct = (current_price - avg_entry_price) / avg_entry_price
                    else:
                        price_change_pct = (avg_entry_price - current_price) / avg_entry_price
                    
                    # 更新全局max_profit_pct（用于状态保存）
                    max_profit_pct = peak_pnl_pct
                    
                    # 如果峰值更新，保存状态
                    if peak_updated:
                        logger.info("📈 更新最高盈利: %.2f%%", max_profit_pct * 100)
                        # 更新派生变量以确保一致性
                        update_derived_position_vars()
                        # 更新状态文件
                        positions_data = [position_to_dict(p) for p in positions]
                        save_trading_state({
                            'version': 2,
                            'positions': positions_data,
                            'last_pyramid_time': str(last_pyramid_time) if last_pyramid_time else None,
                            'open_position_side': open_position_side,
                            'open_entry_time': str(open_entry_time),
                            'predicted_holding_period': predicted_holding_period,
                            'max_profit_pct': max_profit_pct,
                            'open_exposure': open_exposure,
                            'open_entry_price': open_entry_price,
                            'open_position_qty': open_position_qty,
                            'timestamp': str(pd.Timestamp.now(tz='UTC'))
                        }, logger)
                    
                    # 执行平仓
                    if should_close:
                        side = "SELL" if open_position_side == "long" else "BUY"
                        position_side = "LONG" if open_position_side == "long" else "SHORT"
                        
                        logger.info("📤 平仓 %s, 数量=%.4f, 原因=%s, 盈亏=%.2f%% (价格变化%.2f%%)",
                                   open_position_side, open_position_qty, close_reason,
                                   current_pnl_pct * 100, price_change_pct * 100)
                        
                        order_res = client.place_market_order(
                            symbol, side, position_side, open_position_qty, reduce_only=True
                        )
                        
                        if order_res.success:
                            logger.info("✅ 平仓成功: %s", order_res.raw)
                            
                            # 计算盈亏（与回测一致）
                            pnl = calculate_pnl(position, current_price, trading_state.equity)
                            
                            # 更新交易状态（权益、连续亏损计数、风控暂停）
                            update_trading_state(
                                trading_state=trading_state,
                                pnl=pnl,
                                current_time=pd.Timestamp.now(tz='UTC'),
                                max_daily_loss_pct=max_daily_loss_pct,
                                max_drawdown_pause=max_drawdown_pause
                            )
                            
                            # 同步旧变量（用于兼容性）
                            consecutive_losses = trading_state.consecutive_losses
                            daily_loss_paused = trading_state.daily_loss_paused
                            drawdown_paused = trading_state.drawdown_paused
                            peak_equity = trading_state.peak_equity
                            daily_start_balance = trading_state.daily_start_equity
                            
                            # 发送平仓邮件通知
                            if email_notifier:
                                try:
                                    email_notifier.notify_close_position(
                                        side=open_position_side,
                                        quantity=open_position_qty,
                                        entry_price=open_entry_price,
                                        exit_price=current_price,
                                        pnl=pnl,
                                        pnl_pct=current_pnl_pct * 100,
                                        reason=close_reason,
                                        balance=trading_state.equity
                                    )
                                except Exception as e:
                                    logger.warning("平仓邮件通知失败: %s", e)
                            
                            # 重置持仓状态（清空仓位列表）
                            positions.clear()
                            last_pyramid_time = None
                            # 更新派生变量
                            update_derived_position_vars()
                            
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
            
            # 加仓逻辑（金字塔加仓）
            if pyramid_enabled and enable_trading and is_new_bar and should_trade and len(positions) > 0 and len(positions) < pyramid_max_count and not daily_loss_paused and not drawdown_paused:
                try:
                    # 计算当前总盈亏百分比（重用平仓逻辑中的计算，但这里需要重新计算或存储）
                    # 由于平仓逻辑中已经计算了total_pnl_pct，但那是针对所有仓位的，我们可以重新计算
                    # 为了简化，我们重新计算总盈亏百分比
                    total_pnl_pct = 0.0
                    total_exposure = 0.0
                    for pos in positions:
                        if pos.side == 'long':
                            price_change_pct = (current_price - pos.entry_price) / pos.entry_price
                        else:
                            price_change_pct = (pos.entry_price - current_price) / pos.entry_price
                        pnl_pct = price_change_pct * pos.exposure
                        total_pnl_pct += pnl_pct
                        total_exposure += pos.exposure
                    
                    # 计算当前回撤（基于交易状态）
                    if trading_state.peak_equity > 0:
                        current_drawdown = (trading_state.peak_equity - trading_state.equity) / trading_state.peak_equity
                    else:
                        current_drawdown = 0
                    
                    # 计算当前信号敞口
                    new_exposure = calculate_dynamic_exposure(
                        predicted_rr=predicted_rr,
                        direction_prob=direction_prob,
                        current_drawdown=current_drawdown,
                        consecutive_losses=trading_state.consecutive_losses,
                        max_exposure=max_exposure
                    )
                    
                    # 加仓条件检查
                    can_pyramid = (
                        total_pnl_pct > pyramid_profit_threshold and
                        direction == (1 if positions[0].side == 'long' else -1) and
                        predicted_rr >= pyramid_min_rr and
                        direction_prob >= pyramid_min_prob and
                        (last_pyramid_time is None or (current_close_time - last_pyramid_time).total_seconds() >= pyramid_min_bars * 15 * 60) and
                        total_exposure + new_exposure <= max_total_exposure
                    )
                    
                    if can_pyramid:
                        # 计算加仓数量
                        notional_value = trading_state.equity * new_exposure
                        quantity = notional_value / current_price
                        quantity = float(int(quantity * 1000) / 1000)
                        
                        if quantity > 0:
                            desired_side = "long" if direction == 1 else "short"
                            side = "BUY" if desired_side == "long" else "SELL"
                            position_side = "LONG" if desired_side == "long" else "SHORT"
                            
                            logger.info("📥 加仓 %s, 数量=%.4f, 敞口=%.2f倍, 总敞口=%.2f倍, RR=%.2f, prob=%.3f",
                                       desired_side, quantity, new_exposure, total_exposure + new_exposure, predicted_rr, direction_prob)
                            
                            order_res = client.place_market_order(symbol, side, position_side, quantity)
                            
                            if order_res.success:
                                logger.info("✅ 加仓成功: %s", order_res.raw)
                                # 创建新仓位并添加到列表
                                new_position = Position(
                                    side=desired_side,
                                    entry_price=current_price,
                                    entry_time=current_close_time,
                                    exposure=new_exposure,
                                    hold_period=positions[0].hold_period,  # 继承首仓周期
                                    quantity=quantity,
                                    peak_pnl_pct=0.0,
                                    peak_price=0.0
                                )
                                positions.append(new_position)
                                last_pyramid_time = current_close_time
                                # 更新派生变量
                                update_derived_position_vars()
                                
                                # 发送加仓邮件通知
                                if email_notifier:
                                    try:
                                        email_notifier.notify_pyramid_position(
                                            side=desired_side,
                                            quantity=quantity,
                                            price=current_price,
                                            exposure=new_exposure,
                                            total_exposure=total_exposure + new_exposure,
                                            pyramid_count=len(positions),
                                            rr=predicted_rr,
                                            prob=direction_prob,
                                            balance=trading_state.equity
                                        )
                                    except Exception as e:
                                        logger.warning("加仓邮件通知失败: %s", e)
                                
                                # 保存状态
                                positions_data = [position_to_dict(p) for p in positions]
                                save_trading_state({
                                    'version': 2,
                                    'positions': positions_data,
                                    'last_pyramid_time': str(last_pyramid_time) if last_pyramid_time else None,
                                    'open_position_side': open_position_side,
                                    'open_entry_time': str(open_entry_time),
                                    'predicted_holding_period': predicted_holding_period,
                                    'max_profit_pct': max_profit_pct,
                                    'open_exposure': open_exposure,
                                    'open_entry_price': open_entry_price,
                                    'open_position_qty': open_position_qty,
                                    'timestamp': str(pd.Timestamp.now(tz='UTC'))
                                }, logger)
                            else:
                                logger.error("❌ 加仓失败: %s", order_res.raw)
                except Exception as e:
                    logger.exception("加仓逻辑异常: %s", e)
            
            # 开仓逻辑
            if enable_trading and is_new_bar and should_trade and len(positions) == 0 and not daily_loss_paused and not drawdown_paused:
                try:
                    current_balance = client.get_account_balance_usdt()
                    
                    # 更新统一交易状态
                    trading_state.equity = current_balance
                    if current_balance > trading_state.peak_equity:
                        trading_state.peak_equity = current_balance
                    
                    # 计算当前回撤
                    current_drawdown = (trading_state.peak_equity - trading_state.equity) / trading_state.peak_equity if trading_state.peak_equity else 0
                    
                    # 使用共享模块判断是否应该开仓
                    if should_open_position(
                        trading_state=trading_state,
                        should_trade=should_trade,
                        current_drawdown=current_drawdown,
                        max_drawdown_pause=max_drawdown_pause
                    ):
                        # 计算动态敞口
                        optimal_exposure = calculate_dynamic_exposure(
                            predicted_rr=predicted_rr,
                            direction_prob=direction_prob,
                            current_drawdown=current_drawdown,
                            consecutive_losses=trading_state.consecutive_losses,
                            max_exposure=max_exposure
                        )
                        
                        # 计算开仓数量
                        # 敞口 = 杠杆 × 仓位比例
                        # 这里简化为：名义价值 = 余额 × 敞口
                        notional_value = trading_state.equity * optimal_exposure
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
                                
                                # 创建Position对象并添加到仓位列表
                                new_position = Position(
                                    side=desired_side,
                                    entry_price=current_price,
                                    entry_time=current_close_time,
                                    exposure=optimal_exposure,
                                    hold_period=int(holding_period),
                                    quantity=quantity,
                                    peak_pnl_pct=0.0,
                                    peak_price=0.0
                                )
                                positions.append(new_position)
                                last_pyramid_time = current_close_time  # 记录开仓时间（也视为加仓时间）
                                
                                # 更新派生变量
                                update_derived_position_vars()
                                
                                # 发送开仓邮件通知（使用派生变量）
                                if email_notifier:
                                    try:
                                        email_notifier.notify_open_position(
                                            side=open_position_side,
                                            quantity=open_position_qty,
                                            price=open_entry_price,
                                            exposure=open_exposure,
                                            rr=predicted_rr,
                                            prob=direction_prob,
                                            balance=trading_state.equity
                                        )
                                    except Exception as e:
                                        logger.warning("开仓邮件通知失败: %s", e)
                                
                                # 保存状态
                                positions_data = [position_to_dict(p) for p in positions]
                                save_trading_state({
                                    'version': 2,
                                    'positions': positions_data,
                                    'last_pyramid_time': str(last_pyramid_time) if last_pyramid_time else None,
                                    'open_position_side': open_position_side,
                                    'open_entry_time': str(open_entry_time),
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
            
            logger.info("💼 持仓=%s, 余额=%.2f, 每日亏损暂停=%s, 回撤暂停=%s", open_position_side, 
                       current_balance if enable_trading else 0, daily_loss_paused, drawdown_paused)
        
        except Exception as e:
            logger.exception("主循环异常: %s", e)
        
        time.sleep(poll_interval)


if __name__ == '__main__':
    main()
