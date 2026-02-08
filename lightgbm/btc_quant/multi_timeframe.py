"""多时间周期特征与信号融合模块。

核心思路：
1. 同时分析 5m、15m、1h 三个时间周期
2. 每个周期独立计算技术指标
3. 融合不同周期的信号强度
4. 长周期确定趋势，短周期寻找入场点
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import Config


@dataclass
class MultiTimeframeFeatures:
    """多时间周期特征集。"""
    
    # 基础K线数据（以15m为基准）
    df_5m: pd.DataFrame
    df_15m: pd.DataFrame
    df_1h: pd.DataFrame
    
    # 合并后的特征矩阵
    features: pd.DataFrame
    
    # 时间索引（15m级别）
    times: pd.Series


def resample_to_higher_timeframe(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    """将低频K线重采样到高频。
    
    Args:
        df: 原始K线数据（需要有datetime索引）
        target_interval: 目标周期，如 '1h', '4h'
    
    Returns:
        重采样后的DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame必须有DatetimeIndex")
    
    # OHLCV重采样规则
    resample_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    
    # 只保留存在的列
    resample_dict = {k: v for k, v in resample_dict.items() if k in df.columns}
    
    df_resampled = df.resample(target_interval).agg(resample_dict).dropna()
    
    return df_resampled


def calculate_timeframe_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """计算单个时间周期的技术指标。
    
    Args:
        df: K线数据（包含 open, high, low, close, volume）
        prefix: 特征前缀（如 '5m_', '15m_', '1h_'）
    
    Returns:
        特征DataFrame
    """
    features = pd.DataFrame(index=df.index)
    
    # 1. 趋势指标
    features[f'{prefix}ma_7'] = df['close'].rolling(window=7).mean()
    features[f'{prefix}ma_25'] = df['close'].rolling(window=25).mean()
    features[f'{prefix}ma_99'] = df['close'].rolling(window=99).mean()
    features[f'{prefix}ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    features[f'{prefix}ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # 2. 动量指标
    features[f'{prefix}rsi_14'] = _calculate_rsi(df['close'], 14)
    features[f'{prefix}rsi_7'] = _calculate_rsi(df['close'], 7)
    
    # 3. MACD
    macd_line, signal_line, histogram = _calculate_macd(df['close'])
    features[f'{prefix}macd'] = macd_line
    features[f'{prefix}macd_signal'] = signal_line
    features[f'{prefix}macd_hist'] = histogram
    
    # 4. 布林带
    bb_upper, bb_middle, bb_lower = _calculate_bollinger(df['close'], 20)
    features[f'{prefix}bb_width'] = (bb_upper - bb_lower) / bb_middle
    features[f'{prefix}bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)
    
    # 5. ATR波动率
    atr = _calculate_atr(df, 14)
    features[f'{prefix}atr_pct'] = atr / df['close']
    
    # 6. 成交量
    features[f'{prefix}volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # 7. 价格位置（距离高低点的百分比）
    high_20 = df['high'].rolling(window=20).max()
    low_20 = df['low'].rolling(window=20).min()
    features[f'{prefix}price_position'] = (df['close'] - low_20) / (high_20 - low_20 + 1e-9)
    
    # 8. 趋势强度
    features[f'{prefix}trend_strength'] = (features[f'{prefix}ma_7'] - features[f'{prefix}ma_25']) / features[f'{prefix}ma_25']
    
    return features


def _calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    """计算RSI指标。"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """计算MACD指标。"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _calculate_bollinger(series: pd.Series, window: int, num_std: float = 2.0):
    """计算布林带。"""
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, ma, lower


def _calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """计算ATR（平均真实波幅）。"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def build_multi_timeframe_features(
    df_15m: pd.DataFrame,
    cfg: Config,
) -> MultiTimeframeFeatures:
    """构建多时间周期特征。
    
    工作流程：
    1. 从15m K线重采样生成1h数据
    2. 下采样生成5m数据（如果有的话）
    3. 分别计算每个周期的技术指标
    4. 将高周期指标对齐到15m基准
    5. 合并所有特征
    
    Args:
        df_15m: 15分钟K线数据（基准）
        cfg: 配置对象
    
    Returns:
        多时间周期特征集
    """
    # 确保有datetime索引
    if 'open_time' in df_15m.columns:
        df_15m = df_15m.set_index('open_time')
    
    if not isinstance(df_15m.index, pd.DatetimeIndex):
        df_15m.index = pd.to_datetime(df_15m.index, unit='ms')
    
    # 1. 生成1小时数据
    df_1h = resample_to_higher_timeframe(df_15m, '1h')
    
    # 2. 计算各周期特征
    features_15m = calculate_timeframe_features(df_15m, '15m_')
    features_1h = calculate_timeframe_features(df_1h, '1h_')
    
    # 3. 将1h特征对齐到15m（前向填充）
    features_1h_aligned = features_1h.reindex(df_15m.index, method='ffill')
    
    # 4. 合并特征
    all_features = pd.concat([features_15m, features_1h_aligned], axis=1)
    
    # 5. 添加跨周期特征（周期间的关系）
    all_features['ma_divergence_15m_1h'] = (
        features_15m['15m_ma_7'] - features_1h_aligned['1h_ma_7']
    ) / features_1h_aligned['1h_ma_7']
    
    all_features['rsi_diff_15m_1h'] = features_15m['15m_rsi_14'] - features_1h_aligned['1h_rsi_14']
    
    all_features['trend_alignment'] = np.sign(features_15m['15m_trend_strength']) == np.sign(
        features_1h_aligned['1h_trend_strength']
    )
    
    # 删除NaN行
    all_features = all_features.dropna()
    
    # 提取对齐后的时间索引
    times = pd.Series(all_features.index, index=all_features.index)
    
    return MultiTimeframeFeatures(
        df_5m=pd.DataFrame(),  # 暂时不用5m
        df_15m=df_15m.loc[all_features.index],
        df_1h=df_1h,
        features=all_features,
        times=times,
    )


def calculate_multi_timeframe_signal_strength(
    features_row: pd.Series,
) -> Dict[str, float]:
    """基于多时间周期特征计算信号强度。
    
    返回：
    - long_strength: 做多信号强度 (0-1)
    - short_strength: 做空信号强度 (0-1)
    - confidence: 信号置信度 (0-1)
    - trend_filter: 是否通过趋势过滤 (True/False)
    """
    strengths = {}
    
    # 1. 1h趋势判断（主导方向）
    h1_trend = 0.0
    if '1h_ma_7' in features_row.index and '1h_ma_25' in features_row.index:
        if features_row['1h_ma_7'] > features_row['1h_ma_25']:
            h1_trend = min((features_row['1h_ma_7'] / features_row['1h_ma_25'] - 1) * 50, 1.0)
        else:
            h1_trend = max((features_row['1h_ma_7'] / features_row['1h_ma_25'] - 1) * 50, -1.0)
    
    # 2. 15m短期信号
    m15_signal = 0.0
    if '15m_rsi_14' in features_row.index:
        rsi = features_row['15m_rsi_14']
        if rsi < 30:
            m15_signal = (30 - rsi) / 30  # 超卖
        elif rsi > 70:
            m15_signal = -(rsi - 70) / 30  # 超买
    
    # 3. MACD信号
    macd_signal = 0.0
    if '15m_macd_hist' in features_row.index:
        hist = features_row['15m_macd_hist']
        macd_signal = np.tanh(hist * 100)  # 归一化到[-1, 1]
    
    # 4. 布林带位置
    bb_signal = 0.0
    if '15m_bb_position' in features_row.index:
        pos = features_row['15m_bb_position']
        if pos < 0.2:
            bb_signal = 0.5  # 接近下轨
        elif pos > 0.8:
            bb_signal = -0.5  # 接近上轨
    
    # 5. 趋势一致性检查
    trend_aligned = features_row.get('trend_alignment', False)
    
    # 综合信号
    combined_signal = (
        h1_trend * 0.4 +  # 1h趋势权重40%
        m15_signal * 0.2 +  # 15m RSI权重20%
        macd_signal * 0.3 +  # MACD权重30%
        bb_signal * 0.1  # 布林带权重10%
    )
    
    # 做多/做空强度
    long_strength = max(0, combined_signal)
    short_strength = max(0, -combined_signal)
    
    # 置信度：趋势对齐时置信度高
    confidence = 0.7 if trend_aligned else 0.4
    
    # 波动率调整：低波动率时降低置信度
    if '15m_atr_pct' in features_row.index:
        atr_pct = features_row['15m_atr_pct']
        if atr_pct < 0.002:  # 0.2%以下波动率太低
            confidence *= 0.6
    
    # 趋势过滤：只在明确趋势中交易
    trend_filter = abs(h1_trend) > 0.15 and trend_aligned
    
    return {
        'long_strength': float(long_strength),
        'short_strength': float(short_strength),
        'confidence': float(confidence),
        'trend_filter': bool(trend_filter),
    }
