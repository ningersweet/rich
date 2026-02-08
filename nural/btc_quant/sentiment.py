"""市场情绪与深度分析模块。

功能：
1. 资金流向分析（买卖压力）
2. 持仓量（Open Interest）变化
3. 融资费率（Funding Rate）
4. 大单异常检测
5. 市场情绪综合评分
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

from .config import Config


@dataclass
class MarketSentiment:
    """市场情绪数据。"""
    
    # 资金流向（正为流入，负为流出）
    money_flow: float
    
    # 持仓量变化率
    open_interest_change: float
    
    # 融资费率
    funding_rate: float
    
    # 综合情绪评分（-1到1，正为看多）
    sentiment_score: float
    
    # 是否异常市场（剧烈波动/单边）
    is_abnormal: bool


def calculate_money_flow(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算资金流向指标（基于价格和成交量）。
    
    Money Flow = Typical Price × Volume
    MFI (Money Flow Index) = 100 - (100 / (1 + Money Flow Ratio))
    
    Args:
        df: K线数据（需要有high, low, close, volume）
        period: 计算周期
    
    Returns:
        资金流向指标 (-1到1)
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    # 区分正负资金流
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()
    
    # MFI指标
    mfi = 100 - (100 / (1 + positive_sum / (negative_sum + 1e-9)))
    
    # 归一化到[-1, 1]
    normalized = (mfi - 50) / 50
    
    return normalized


def calculate_volume_profile(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """计算成交量分布异常。
    
    检测：
    - 放量上涨/下跌
    - 缩量整理
    
    Returns:
        异常评分（-1到1）
    """
    volume_ma = df['volume'].rolling(window=lookback).mean()
    volume_std = df['volume'].rolling(window=lookback).std()
    
    volume_zscore = (df['volume'] - volume_ma) / (volume_std + 1e-9)
    
    # 价格变化
    price_change = df['close'].pct_change()
    
    # 放量上涨：正信号
    # 放量下跌：负信号
    signal = np.sign(price_change) * np.clip(volume_zscore / 3, -1, 1)
    
    return signal


def fetch_funding_rate(symbol: str = "BTCUSDT") -> Optional[float]:
    """从币安获取当前融资费率。
    
    融资费率含义：
    - 正值：多头支付空头（市场看多过热）
    - 负值：空头支付多头（市场看空过热）
    
    Returns:
        融资费率（小数形式，如0.0001代表0.01%）
    """
    try:
        url = "https://fapi.binance.com/fapi/v1/premiumIndex"
        params = {"symbol": symbol}
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        return float(data.get('lastFundingRate', 0))
    except Exception:
        # 如果获取失败，返回None（可以用历史数据估算）
        return None


def fetch_open_interest(symbol: str = "BTCUSDT") -> Optional[float]:
    """从币安获取当前持仓量（Open Interest）。
    
    持仓量含义：
    - 增加：新资金进场
    - 减少：资金离场
    
    Returns:
        持仓量（USDT计价）
    """
    try:
        url = "https://fapi.binance.com/fapi/v1/openInterest"
        params = {"symbol": symbol}
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        # 返回USDT计价的持仓量
        return float(data.get('openInterest', 0)) * float(data.get('markPrice', 0))
    except Exception:
        return None


def calculate_market_sentiment(
    df: pd.DataFrame,
    current_funding_rate: Optional[float] = None,
    current_oi: Optional[float] = None,
    previous_oi: Optional[float] = None,
) -> MarketSentiment:
    """计算综合市场情绪。
    
    Args:
        df: 最近的K线数据（至少20根）
        current_funding_rate: 当前融资费率
        current_oi: 当前持仓量
        previous_oi: 前一时刻持仓量
    
    Returns:
        市场情绪对象
    """
    # 1. 资金流向
    money_flow_series = calculate_money_flow(df, period=14)
    money_flow = float(money_flow_series.iloc[-1]) if not money_flow_series.empty else 0.0
    
    # 2. 成交量异常
    volume_signal_series = calculate_volume_profile(df, lookback=20)
    volume_signal = float(volume_signal_series.iloc[-1]) if not volume_signal_series.empty else 0.0
    
    # 3. 融资费率信号
    funding_signal = 0.0
    if current_funding_rate is not None:
        # 融资费率通常在-0.001到0.001之间
        # 过高的正费率（>0.0005）说明多头过热，反向信号
        # 过低的负费率（<-0.0005）说明空头过热，反向信号
        if abs(current_funding_rate) > 0.0005:
            funding_signal = -np.sign(current_funding_rate) * 0.5  # 反向信号
    
    # 4. 持仓量变化
    oi_change = 0.0
    if current_oi is not None and previous_oi is not None and previous_oi > 0:
        oi_change = (current_oi - previous_oi) / previous_oi
        # 持仓量大幅增加（>5%）可能预示趋势加速
        # 持仓量大幅减少（<-5%）可能预示趋势反转
    
    # 5. 综合情绪评分
    sentiment_score = (
        money_flow * 0.4 +  # 资金流向权重40%
        volume_signal * 0.3 +  # 成交量权重30%
        funding_signal * 0.2 +  # 融资费率权重20%
        np.sign(oi_change) * min(abs(oi_change) * 10, 0.5) * 0.1  # 持仓量变化权重10%
    )
    
    sentiment_score = np.clip(sentiment_score, -1, 1)
    
    # 6. 异常市场检测
    is_abnormal = False
    if len(df) >= 5:
        recent_volatility = df['close'].pct_change().tail(5).std()
        if recent_volatility > 0.03:  # 3%以上波动率
            is_abnormal = True
        
        # 检测单边行情（连续5根同向）
        recent_changes = df['close'].pct_change().tail(5)
        if all(recent_changes > 0) or all(recent_changes < 0):
            is_abnormal = True
    
    return MarketSentiment(
        money_flow=float(money_flow),
        open_interest_change=float(oi_change),
        funding_rate=float(current_funding_rate) if current_funding_rate is not None else 0.0,
        sentiment_score=float(sentiment_score),
        is_abnormal=bool(is_abnormal),
    )


def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """为K线数据添加情绪特征列。
    
    Args:
        df: K线数据
    
    Returns:
        添加了情绪特征的DataFrame
    """
    df = df.copy()
    
    # 资金流向
    df['money_flow'] = calculate_money_flow(df, period=14)
    
    # 成交量异常
    df['volume_signal'] = calculate_volume_profile(df, lookback=20)
    
    # 价格动量
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    
    # 波动率
    df['volatility_5'] = df['close'].pct_change().rolling(window=5).std()
    df['volatility_20'] = df['close'].pct_change().rolling(window=20).std()
    
    # 价格-成交量相关性
    price_change = df['close'].pct_change()
    volume_change = df['volume'].pct_change()
    df['price_volume_corr'] = price_change.rolling(window=10).corr(volume_change)
    
    return df
