from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from .config import Config


@dataclass
class FeatureLabelData:
    features: pd.DataFrame
    labels: pd.Series


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


def _bollinger_bands(series: pd.Series, window: int, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """计算布林带"""
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, ma, lower


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """计算MACD指标"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _stochastic(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """计算随机指标KDJ中的K值"""
    low_min = df["low"].rolling(window=window).min()
    high_max = df["high"].rolling(window=window).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-9)
    return k


def _volume_profile(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """成交量相对均值的比率"""
    vol_ma = df["volume"].rolling(window=window).mean()
    return df["volume"] / (vol_ma + 1e-9)


def _momentum(series: pd.Series, window: int) -> pd.Series:
    """动量指标：当前价格相对N期前的涨跌幅"""
    return series.pct_change(window)


def _market_regime(df: pd.DataFrame) -> pd.Series:
    """市场状态分类：牛市(1)/震荡(0)/熊市(-1)
    
    基于价格相对长期均线的位置和趋势强度
    """
    # 使用99周期均线作为长期趋势参考
    ma_long = df["close"].rolling(window=99).mean()
    # 价格相对均线的位置
    price_vs_ma = (df["close"] - ma_long) / (ma_long + 1e-9)
    
    # 计算近期趋势强度（20周期收益率）
    trend_strength = df["close"].pct_change(20)
    
    # 分类逻辑：
    # 牛市：价格在长期均线之上且趋势向上
    # 熊市：价格在长期均线之下且趋势向下
    # 震荡：其他情况
    regime = pd.Series(0, index=df.index)  # 默认震荡
    
    # 牛市条件
    bull_mask = (price_vs_ma > 0.02) & (trend_strength > 0.05)
    regime[bull_mask] = 1
    
    # 熊市条件
    bear_mask = (price_vs_ma < -0.02) & (trend_strength < -0.05)
    regime[bear_mask] = -1
    
    return regime


def build_features_and_labels(cfg: Config, klines: pd.DataFrame) -> FeatureLabelData:
    """根据配置从 K 线数据构建特征与标签。

    特征：多维度技术指标 + 动量 + 波动率 + 成交量等。
    标签：基于未来 N 根K线收益率的多/空/观望三分类。
    """

    df = klines.copy().reset_index(drop=True)

    # === 1. 基本价格与收益率 ===
    df["return_1"] = df["close"].pct_change().astype("float32")
    df["return_3"] = df["close"].pct_change(3).astype("float32")
    df["return_7"] = df["close"].pct_change(7).astype("float32")
    
    # 高低价差率
    df["hl_ratio"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)
    
    # 开收盘幅度
    df["oc_ratio"] = (df["close"] - df["open"]) / (df["open"] + 1e-9)

    # === 2. 多周期均线特征 ===
    ma_windows = cfg.features.get("ma_windows", [7, 25, 99])
    for w in ma_windows:
        df[f"ma_{w}"] = df["close"].rolling(window=w).mean()
        df[f"ema_{w}"] = _ema(df["close"], span=w)
        # 价格相对均线的离差率
        df[f"close_ma_{w}_diff"] = (df["close"] - df[f"ma_{w}"]) / (df[f"ma_{w}"] + 1e-9)
        
        # === 新增：均线斜率（方案B） ===
        # 对于超长周期均线，使用斜率而非绝对值
        if w >= 50:
            df[f"ma_{w}_slope_5"] = (df[f"ma_{w}"] - df[f"ma_{w}"].shift(5)) / (df[f"ma_{w}"].shift(5) + 1e-9)
    
    # 均线之间的关系（金叉/死叉）
    if len(ma_windows) >= 2:
        df[f"ma_{ma_windows[0]}_ma_{ma_windows[1]}_ratio"] = df[f"ma_{ma_windows[0]}"] / (df[f"ma_{ma_windows[1]}"] + 1e-9)

    # === 3. RSI 与 ATR ===
    rsi_window = int(cfg.features.get("rsi_window", 14))
    atr_window = int(cfg.features.get("atr_window", 14))
    df[f"rsi_{rsi_window}"] = _rsi(df["close"], rsi_window)
    
    # === 新增：短周期RSI（方案B） ===
    df["rsi_6"] = _rsi(df["close"], 6)  # 短周期超买超卖
    
    df[f"atr_{atr_window}"] = _atr(df, atr_window)
    # ATR 占价格的比例，用于波动率过滤
    df["atr_pct"] = df[f"atr_{atr_window}"] / df["close"]
    
    # ATR的变化趋势（波动率是否扩张）
    df["atr_change"] = df[f"atr_{atr_window}"].pct_change(5)

    # === 4. 布林带 ===
    bb_window = 20
    bb_upper, bb_mid, bb_lower = _bollinger_bands(df["close"], bb_window)
    df["bb_upper"] = bb_upper
    df["bb_middle"] = bb_mid  # 保存中轨
    df["bb_lower"] = bb_lower
    df["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-9)  # 布林带宽度
    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)  # 价格在布林带中的位置

    # === 5. MACD ===
    macd_line, signal_line, histogram = _macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = histogram
    df["macd_hist_change"] = histogram.diff()  # MACD柱的变化

    # === 6. KDJ/随机指标 ===
    df["stoch_k"] = _stochastic(df, window=14)
    df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()  # D线是K线的3日均线

    # === 7. 动量指标 ===
    for mom_window in [5, 10, 20]:
        df[f"momentum_{mom_window}"] = _momentum(df["close"], mom_window)

    # === 8. 成交量特征 ===
    df["volume_ratio"] = _volume_profile(df, window=20)
    df["volume_ma_5"] = df["volume"].rolling(window=5).mean()
    df["volume_change"] = df["volume"].pct_change()
    
    # === 新增：成交量突增特征（方案B） ===
    df["volume_spike_ratio"] = df["volume"] / (df["volume"].rolling(window=20).mean() + 1e-9)
    
    # 价量协同：价格涨且成交量放大
    df["price_volume_corr"] = df["return_1"] * df["volume_change"]

    # === 9. 波动率regime特征 ===
    # 近期波动率相对长期波动率
    short_vol = df["return_1"].rolling(window=10).std()
    long_vol = df["return_1"].rolling(window=50).std()
    df["volatility_regime"] = short_vol / (long_vol + 1e-9)

    # === 10. 价格位置特征 ===
    window_pos = max(ma_windows)
    rolling_max = df["high"].rolling(window=window_pos).max()
    rolling_min = df["low"].rolling(window=window_pos).min()
    df["price_position"] = (df["close"] - rolling_min) / (rolling_max - rolling_min + 1e-9)
    
    # === 新增：边界特征（方案B） ===
    # 距离最近20根K线高点的百分比
    recent_high_20 = df["high"].rolling(window=20).max()
    df["dist_to_recent_high_20"] = (recent_high_20 - df["close"]) / (df["close"] + 1e-9)
    
    # 距离最近20根K线低点的百分比
    recent_low_20 = df["low"].rolling(window=20).min()
    df["dist_to_recent_low_20"] = (df["close"] - recent_low_20) / (df["close"] + 1e-9)
    
    # 价格在布林带中的位置（0-1）
    df["price_in_bband_percentile"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    
    # 布林带宽度百分比（波动率挤压指标）
    df["bb_width_pct"] = df["bb_width"] / (df["bb_middle"] + 1e-9)
    
    # 距离近期高/低点的距离
    df["dist_to_high"] = (rolling_max - df["close"]) / (df["close"] + 1e-9)
    df["dist_to_low"] = (df["close"] - rolling_min) / (df["close"] + 1e-9)

    # === 11. 市场状态regime ===
    df["market_regime"] = _market_regime(df)
    
    # === 12. DeepSeek熊市专用特征（关键！） ===
    
    # 特征1：反弹强度（K线实体相对波动率）
    # 衡量单根K线的反弹/下跌力度
    df["rebound_strength"] = (df["high"] - df["low"]) / (df[f"atr_{atr_window}"] + 1e-9)
    
    # 特征2：超卖修复（短期RSI是否从超卖修复）
    # 正值：短期RSI高于中期RSI，超卖修复中
    # 负值：短期RSI低于中期RSI，超买修复中
    df["oversold_recovery"] = df["rsi_6"] - df[f"rsi_{rsi_window}"]
    
    # 特征3：空头陷阱（假突破检测）
    # 价格创新低但收盘价高于前一根收盘价，可能是假突破
    previous_low = df["low"].shift(1)
    previous_close = df["close"].shift(1)
    is_bear_trap = (df["low"] < previous_low) & (df["close"] > previous_close)
    df["bear_trap"] = is_bear_trap.astype("float32")
    
    # 特征4：量价背离（成交量与价格方向不一致）
    # 正值：价格下跌但成交量放大（看涨背离）
    # 负值：价格上涨但成交量放大（看跌背离）
    price_direction = df["close"] - df["open"]  # 正值=收阳，负值=收阴
    volume_weighted = df["volume"] / (df["volume"].rolling(window=20).mean() + 1e-9)
    df["volume_price_divergence"] = price_direction * volume_weighted / (df["close"] + 1e-9)
    
    # 特征5：关键支撑距离（距离最近20根K线低点的ATR倍数）
    # 小值：离支撑位很近，可能反弹
    # 大值：离支撑位很远，可能继续下跌
    support_level = df["low"].rolling(window=20).min()
    df["dist_to_support_atr"] = (df["close"] - support_level) / (df[f"atr_{atr_window}"] + 1e-9)

    # ===== 标签构建：突破标签（DeepSeek方案B） =====
    labeling_cfg = cfg.labeling
    horizon = int(labeling_cfg.get("horizon_bars", 10))
    label_method = labeling_cfg.get("method", "future_return")

    label = np.zeros(len(df), dtype="int8")  # 0: 观望

    if label_method == "breakthrough":
        # DeepSeek熊市优化：改进的突破标签逻辑
        # 核心改进：
        # 1. 做多标签要求更严格（2.0倍ATR，需要强反弹）
        # 2. 做空标签更智能（1.0倍ATR，但排除反弹起点）
        atr_multiplier_long = float(labeling_cfg.get("atr_multiplier_long", 2.0))
        atr_multiplier_short = float(labeling_cfg.get("atr_multiplier_short", 1.0))
        require_trend = labeling_cfg.get("require_trend_confirmation", False)
        
        # 计算未来窗口内的最高价和最低价
        future_high = df["high"].rolling(window=horizon).max().shift(-horizon)
        future_low = df["low"].rolling(window=horizon).min().shift(-horizon)
        
        # 动态阈值：做多更严格，做空更宽松
        threshold_pct_long = df["atr_pct"] * atr_multiplier_long
        threshold_pct_short = df["atr_pct"] * atr_multiplier_short
        
        # === 做多标签：未来强反弹 ===
        # 要求：未来窗口内最高价突破当前收盘价 + 2.0倍ATR
        long_breakthrough = future_high > df["close"] * (1 + threshold_pct_long)
        
        # === 做空标签：未来继续下跌，但排除反弹起点 ===
        # 基础条件：未来窗口内最低价跌破当前收盘价 - 1.0倍ATR
        short_breakthrough = future_low < df["close"] * (1 - threshold_pct_short)
        
        # DeepSeek关键优化：排除在反弹起点做空
        if require_trend:
            # 计算MA20（如果不存在）
            if "ma_20" not in df.columns:
                ma_20 = df["close"].rolling(window=20).mean()
            else:
                ma_20 = df["ma_20"]
            
            # 条件1：价格低于MA20（下降趋势）
            in_downtrend = df["close"] < ma_20
            
            # 条件2：不是刚刚创新低后的强反弹（调整：放宽判断）
            # 检测：当前RSI6 > 30（不是极端超卖）或 价格不是在布林带下轨附近
            # 这样可以在熟悉的下降趋势中做空，但避免在极端超卖后的反弹做空
            not_extreme_oversold = df["rsi_6"] > 30
            not_at_lower_band = df["close"] > df["bb_lower"] * 1.005  # 不在下轨以下
            
            # 任意满足一个即可（放宽条件）
            not_bounce_point = not_extreme_oversold | not_at_lower_band
            
            # 同时满足两个条件才做空
            short_breakthrough = short_breakthrough & in_downtrend & not_bounce_point
        
        label[long_breakthrough] = 1   # 做多
        label[short_breakthrough] = -1  # 做空
        
    elif label_method == "triple_barrier":
        # 三重屏障标签：先触及止盈为1，先触及止损为-1
        profit_multiplier = float(labeling_cfg.get("profit_atr_multiplier", 2.0))
        loss_multiplier = float(labeling_cfg.get("loss_atr_multiplier", 1.0))
        
        for i in range(len(df) - horizon):
            current_price = df["close"].iloc[i]
            current_atr = df["atr_14"].iloc[i]
            
            # 止盈和止损价位
            take_profit_long = current_price * (1 + current_atr * profit_multiplier / current_price)
            stop_loss_long = current_price * (1 - current_atr * loss_multiplier / current_price)
            take_profit_short = current_price * (1 - current_atr * profit_multiplier / current_price)
            stop_loss_short = current_price * (1 + current_atr * loss_multiplier / current_price)
            
            # 检查未来窗口内的价格行为
            future_slice = df.iloc[i+1:i+1+horizon]
            
            # 做多方向
            hit_profit_long = (future_slice["high"] >= take_profit_long).any()
            hit_loss_long = (future_slice["low"] <= stop_loss_long).any()
            
            if hit_profit_long and not hit_loss_long:
                label[i] = 1
            elif hit_loss_long:
                # 先触及止损的不标记为做多机会
                pass
            
            # 做空方向
            hit_profit_short = (future_slice["low"] <= take_profit_short).any()
            hit_loss_short = (future_slice["high"] >= stop_loss_short).any()
            
            if hit_profit_short and not hit_loss_short:
                label[i] = -1
    
    else:
        # 旧逻辑：预测固定收益率（保留作为fallback）
        dynamic_threshold = bool(labeling_cfg.get("dynamic_threshold", False))
        future_price = df["close"].shift(-horizon)
        future_ret = (future_price - df["close"]) / df["close"]

        if dynamic_threshold:
            base_pos = float(labeling_cfg.get("base_pos_threshold", labeling_cfg.get("pos_threshold", 0.005)))
            base_neg = float(labeling_cfg.get("base_neg_threshold", labeling_cfg.get("neg_threshold", -0.005)))
            vol_scaling = float(labeling_cfg.get("vol_scaling_factor", 1.5))
            min_th = float(labeling_cfg.get("min_threshold", 0.002))
            max_th = float(labeling_cfg.get("max_threshold", 0.015))

            if "atr_pct" in df.columns:
                vol = df["atr_pct"].abs().clip(lower=1e-6)
            else:
                vol = df["return_1"].rolling(window=20).std().abs()

            baseline = vol.rolling(window=200, min_periods=50).median()
            baseline = baseline.fillna(baseline.median())
            vol_ratio = vol / (baseline + 1e-9)

            pos_th_series = base_pos * (1.0 + vol_scaling * (vol_ratio - 1.0))
            neg_th_series = base_neg * (1.0 + vol_scaling * (vol_ratio - 1.0))

            pos_th_series = pos_th_series.clip(lower=min_th, upper=max_th)
            neg_th_series = neg_th_series.clip(lower=-max_th, upper=-min_th)

            label[future_ret > pos_th_series] = 1
            label[future_ret < neg_th_series] = -1
        else:
            pos_th = float(labeling_cfg.get("pos_threshold", 0.005))
            neg_th = float(labeling_cfg.get("neg_threshold", -0.005))
            label[future_ret > pos_th] = 1
            label[future_ret < neg_th] = -1

    df["label"] = label

    # 去掉前期滚动窗口导致的 NaN，以及末尾未来收益不可计算的行
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        c
        for c in df.columns
        if c
        not in {
            "open_time",
            "close_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "label",
        }
    ]

    X = df[feature_cols].astype("float32")
    y = df["label"].astype("int8")

    return FeatureLabelData(features=X, labels=y)
