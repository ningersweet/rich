#!/usr/bin/env python3
from __future__ import annotations

import time

import pandas as pd

from btc_quant.config import load_config
from btc_quant.data import BINANCE_FAPI_KLINES_ENDPOINT
from btc_quant.execution import BinanceFuturesClient
from btc_quant.features import build_features_and_labels
from btc_quant.model import load_model
from btc_quant.monitor import setup_logger
from btc_quant.position import calculate_position_size
from btc_quant.signals import generate_signal

import requests


def fetch_latest_klines(symbol: str, interval: str, limit: int, base_url: str) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(base_url + BINANCE_FAPI_KLINES_ENDPOINT, params=params, timeout=10)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        raise RuntimeError("未获取到最新K线")
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype("float32")
    return df


def main() -> None:
    cfg = load_config()
    logger = setup_logger(cfg)
    client = BinanceFuturesClient(cfg)

    # 检查当前模式
    mode = cfg.api.get("mode", "paper")
    if mode == "paper":
        logger.warning("当前模式：测试网（paper），使用 API: %s", client.base_url)
        logger.warning("⚠️  将真实下单到币安测试网！")
    else:
        logger.warning("当前模式：实盘（live），使用 API: %s", client.base_url)
        logger.warning("⚠️⚠️⚠️  将真实下单到币安实盘！！！")
    
    # 无论什么模式都启用下单
    enable_trading = True

    symbol = cfg.symbol["name"]
    interval = cfg.symbol["interval"]
    base_url = client.base_url

    trained = load_model(cfg)

    poll_interval = int(cfg.live.get("poll_interval_seconds", 60))
    max_new_bars = int(cfg.live.get("max_new_bars", 500))
    max_daily_loss_pct = float(cfg.risk.get("max_daily_loss_pct", 0.05))

    last_close_time = None
    last_signal: str | None = None
    last_signal_time = None
    current_sig = None  # 当前信号，用于下单
    current_features = None  # 当前特征，用于读取ATR

    starting_balance: float | None = None
    trading_disabled_due_to_loss = False

    # 本地持仓状态（用于控制开平仓逻辑），启动时尝试与交易所同步
    open_position_side: str = "flat"  # flat / long / short
    open_position_qty: float = 0.0
    open_entry_price: float = 0.0

    if enable_trading:
        try:
            starting_balance = client.get_account_balance_usdt()
            pos = client.get_open_position(symbol)
            if pos:
                pos_amt = float(pos.get("positionAmt", 0.0))
                open_entry_price = float(pos.get("entryPrice", 0.0))
                if pos_amt > 0:
                    open_position_side = "long"
                    open_position_qty = abs(pos_amt)
                elif pos_amt < 0:
                    open_position_side = "short"
                    open_position_qty = abs(pos_amt)
            logger.info(
                "初始化账户：起始余额=%.2f, 持仓方向=%s, 持仓数量=%.4f, 开仓价=%.2f",
                starting_balance or 0.0,
                open_position_side,
                open_position_qty,
                open_entry_price,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("初始化账户信息失败: %s", e)

    while True:
        try:
            df = fetch_latest_klines(symbol, interval, max_new_bars, base_url)
            latest = df.iloc[-1:]
            current_close_time = latest["close_time"].iloc[0]
            current_price = float(latest["close"].iloc[0])

            # 检查是否有新K线
            has_new_kline = (last_close_time is None or current_close_time > last_close_time)
            
            if has_new_kline:
                # 有新K线，重新计算特征和预测
                last_close_time = current_close_time
                fl = build_features_and_labels(cfg, df)
                latest_features = fl.features.iloc[[-1]]
                current_features = latest_features  # 保存特征

                sig = generate_signal(
                    cfg,
                    trained,
                    latest_features,
                    prev_signal=last_signal,
                    prev_signal_time=last_signal_time,
                    current_time=current_close_time,
                )
                current_sig = sig  # 保存信号
                if sig.signal != "flat":
                    last_signal = sig.signal
                    last_signal_time = current_close_time

                logger.info(
                    "[新K线] 时间=%s, 价格=%.2f, 信号=%s, prob_long=%.3f, prob_short=%.3f, prob_flat=%.3f",
                    current_close_time,
                    current_price,
                    sig.signal,
                    sig.prob_long,
                    sig.prob_short,
                    sig.prob_flat,
                )
            else:
                # 无新K线，仅输出当前价格和持仓状态
                logger.info(
                    "[轮询] 时间=%s, 价格=%.2f, 持仓=%s",
                    current_close_time,
                    current_price,
                    open_position_side,
                )

            # 仅在启用交易且未触发日内亏损阈值时才尝试下单
            # 并且必须有有效的信号（至少有一次新K线）
            if enable_trading and current_sig is not None:
                try:
                    current_balance = client.get_account_balance_usdt()
                except Exception as e:  # noqa: BLE001
                    logger.exception("获取账户余额失败: %s", e)
                    current_balance = None

                if starting_balance is not None and current_balance is not None:
                    loss_pct = (starting_balance - current_balance) / max(starting_balance, 1e-6)
                    if loss_pct >= max_daily_loss_pct:
                        if not trading_disabled_due_to_loss:
                            logger.error(
                                "日内亏损达到阈值 %.2f%%，当前亏损 %.2f%%，停止开新仓。",
                                max_daily_loss_pct * 100,
                                loss_pct * 100,
                            )
                        trading_disabled_due_to_loss = True

                if not trading_disabled_due_to_loss:
                    price = current_price

                    # 从特征中读取 ATR 供仓位管理使用
                    atr_val = None
                    if current_features is not None:
                        atr_col = f"atr_{cfg.features.get('atr_window', 14)}"
                        if atr_col in current_features.columns:
                            atr_val = float(current_features[atr_col].iloc[0])

                    # 根据信号控制开平仓
                    if current_sig.signal == "flat":
                        # 有持仓则平仓
                        if open_position_side != "flat" and open_position_qty > 0:
                            side = "SELL" if open_position_side == "long" else "BUY"
                            position_side = "LONG" if open_position_side == "long" else "SHORT"
                            logger.info(
                                "平仓 %s, 数量=%.4f", open_position_side, open_position_qty
                            )
                            order_res = client.place_market_order(
                                symbol,
                                side,
                                position_side,
                                open_position_qty,
                                reduce_only=True,
                            )
                            if order_res.success:
                                logger.info("平仓成功: %s", order_res.raw)
                                open_position_side = "flat"
                                open_position_qty = 0.0
                                open_entry_price = 0.0
                            else:
                                logger.error("平仓失败: %s", order_res.raw)
                    else:
                        desired_side = current_sig.signal  # long / short

                        # 若方向相反，先平掉原有持仓
                        if (
                            open_position_side != "flat"
                            and open_position_side != desired_side
                            and open_position_qty > 0
                        ):
                            close_side = "SELL" if open_position_side == "long" else "BUY"
                            close_position_side = "LONG" if open_position_side == "long" else "SHORT"
                            logger.info(
                                "反向信号，先平仓 %s, 数量=%.4f", open_position_side, open_position_qty
                            )
                            order_res = client.place_market_order(
                                symbol,
                                close_side,
                                close_position_side,
                                open_position_qty,
                                reduce_only=True,
                            )
                            if order_res.success:
                                logger.info("平仓成功: %s", order_res.raw)
                                open_position_side = "flat"
                                open_position_qty = 0.0
                                open_entry_price = 0.0
                            else:
                                logger.error("平仓失败: %s", order_res.raw)

                        # 若当前无持仓，则按风险参数开新仓
                        if open_position_side == "flat" and current_balance is not None:
                            pos_info = calculate_position_size(cfg, current_balance, price, atr_val)
                            notional = pos_info.position_usdt
                            if notional <= 0:
                                logger.warning("计算得到名义仓位<=0，跳过下单。")
                            else:
                                qty = notional / price
                                if qty <= 0:
                                    logger.warning("计算得到下单数量<=0，跳过下单。")
                                else:
                                    side = "BUY" if desired_side == "long" else "SELL"
                                    position_side = "LONG" if desired_side == "long" else "SHORT"
                                    logger.info(
                                        "开仓 %s, 价格=%.2f, 名义价值=%.2f, 数量=%.4f, 杠杆=%.2f",
                                        desired_side,
                                        price,
                                        notional,
                                        qty,
                                        pos_info.leverage,
                                    )
                                    order_res = client.place_market_order(
                                        symbol,
                                        side,
                                        position_side,
                                        qty,
                                        reduce_only=False,
                                    )
                                    if order_res.success:
                                        logger.info("开仓成功: %s", order_res.raw)
                                        open_position_side = desired_side
                                        open_position_qty = qty
                                        open_entry_price = price
                                    else:
                                        logger.error("开仓失败: %s", order_res.raw)

        except Exception as e:  # noqa: BLE001
            logger.exception("实盘轮询发生异常: %s", e)

        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
