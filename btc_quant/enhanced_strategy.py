"""增强策略引擎 - 整合多时间周期、市场情绪、集成学习。

这是新策略框架的核心编排模块。

工作流程：
1. 多时间周期特征提取（15m + 1h）
2. 市场情绪分析
3. 集成模型预测
4. 信号置信度综合评估
5. 趋势过滤与风控
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config
from .ensemble import EnsembleModel, EnsemblePrediction, predict_ensemble
from .multi_timeframe import (
    MultiTimeframeFeatures,
    build_multi_timeframe_features,
    calculate_multi_timeframe_signal_strength,
)
from .sentiment import MarketSentiment, calculate_market_sentiment


class SignalLevel(Enum):
    """信号置信度等级。"""
    HIGH = "high"  # 高置信度：模型一致性>0.7，趋势过滤通过
    MEDIUM = "medium"  # 中等置信度：模型一致性0.5-0.7
    LOW = "low"  # 低置信度：模型一致性<0.5
    NONE = "none"  # 无信号


@dataclass
class EnhancedSignal:
    """增强版交易信号。"""
    
    # 信号方向：1做多，-1做空，0观望
    direction: int
    
    # 信号等级
    level: SignalLevel
    
    # 综合置信度（0-1）
    confidence: float
    
    # 做多/做空强度（用于仓位管理）
    long_strength: float
    short_strength: float
    
    # 是否通过趋势过滤
    trend_filter_passed: bool
    
    # 市场情绪评分
    sentiment_score: float
    
    # 模型预测一致性
    model_consistency: float
    
    # 是否异常市场
    is_abnormal_market: bool
    
    # 建议仓位比例（0-1）
    suggested_position_ratio: float


def generate_enhanced_signal(
    df_15m: pd.DataFrame,
    ensemble_model: EnsembleModel,
    cfg: Config,
    current_funding_rate: Optional[float] = None,
    current_oi: Optional[float] = None,
    previous_oi: Optional[float] = None,
) -> EnhancedSignal:
    """生成增强版交易信号。
    
    Args:
        df_15m: 15分钟K线数据（至少100根）
        ensemble_model: 集成模型
        cfg: 配置对象
        current_funding_rate: 当前融资费率
        current_oi: 当前持仓量
        previous_oi: 前一时刻持仓量
    
    Returns:
        增强版交易信号
    """
    # 1. 构建多时间周期特征
    mtf_features = build_multi_timeframe_features(df_15m, cfg)
    
    # 取最后一行特征
    latest_features = mtf_features.features.iloc[[-1]]
    
    # 2. 集成模型预测
    ensemble_pred = predict_ensemble(ensemble_model, latest_features)
    
    # 3. 计算多时间周期信号强度
    mtf_signal = calculate_multi_timeframe_signal_strength(latest_features.iloc[0])
    
    # 4. 计算市场情绪
    recent_df = df_15m.tail(30)  # 最近30根K线
    sentiment = calculate_market_sentiment(
        recent_df,
        current_funding_rate=current_funding_rate,
        current_oi=current_oi,
        previous_oi=previous_oi,
    )
    
    # 5. 综合决策
    
    # 5.1 基础方向判断（基于模型预测）
    prob_long = ensemble_pred.prob_long
    prob_short = ensemble_pred.prob_short
    prob_neutral = ensemble_pred.prob_neutral
    
    # 获取配置参数
    strategy_cfg = cfg.strategy
    min_prob = float(strategy_cfg.get("min_prob", 0.35))
    prob_gap = float(strategy_cfg.get("prob_gap_long", 0.1))
    
    direction = 0
    if prob_long > min_prob and (prob_long - prob_short) > prob_gap:
        direction = 1
    elif prob_short > min_prob and (prob_short - prob_long) > prob_gap:
        direction = -1
    
    # 5.2 趋势过滤（关键！）
    trend_filter_passed = mtf_signal['trend_filter']
    
    # 如果没通过趋势过滤，强制观望
    if not trend_filter_passed:
        direction = 0
    
    # 5.3 异常市场过滤
    if sentiment.is_abnormal:
        # 异常市场降低仓位或观望
        pass  # 后面通过position_ratio控制
    
    # 5.4 情绪与模型方向冲突检测
    sentiment_direction = 1 if sentiment.sentiment_score > 0.3 else (-1 if sentiment.sentiment_score < -0.3 else 0)
    
    sentiment_conflict = False
    if direction != 0 and sentiment_direction != 0:
        if direction != sentiment_direction:
            sentiment_conflict = True
    
    # 5.5 综合置信度计算
    confidence_factors = [
        ensemble_pred.consistency * 0.35,  # 模型一致性35%
        mtf_signal['confidence'] * 0.25,  # 多周期置信度25%
        (1.0 if trend_filter_passed else 0.3) * 0.2,  # 趋势过滤20%
        (abs(sentiment.sentiment_score) if not sentiment_conflict else 0.2) * 0.15,  # 情绪一致性15%
        (0.6 if not sentiment.is_abnormal else 0.3) * 0.05,  # 市场正常性5%
    ]
    
    confidence = sum(confidence_factors)
    confidence = max(0.0, min(1.0, confidence))
    
    # 5.6 信号等级判定
    if confidence >= 0.7 and ensemble_pred.consistency >= 0.7:
        signal_level = SignalLevel.HIGH
    elif confidence >= 0.5:
        signal_level = SignalLevel.MEDIUM
    elif confidence >= 0.3:
        signal_level = SignalLevel.LOW
    else:
        signal_level = SignalLevel.NONE
        direction = 0  # 信号太弱，强制观望
    
    # 5.7 建议仓位比例
    base_position = 1.0
    
    # 根据信号等级调整
    if signal_level == SignalLevel.HIGH:
        base_position = 1.0
    elif signal_level == SignalLevel.MEDIUM:
        base_position = 0.6
    elif signal_level == SignalLevel.LOW:
        base_position = 0.3
    else:
        base_position = 0.0
    
    # 根据模型一致性微调
    base_position *= (0.7 + 0.3 * ensemble_pred.consistency)
    
    # 异常市场降低仓位
    if sentiment.is_abnormal:
        base_position *= 0.5
    
    # 情绪冲突降低仓位
    if sentiment_conflict:
        base_position *= 0.7
    
    suggested_position_ratio = max(0.0, min(1.0, base_position))
    
    # 如果方向为观望，仓位为0
    if direction == 0:
        suggested_position_ratio = 0.0
    
    return EnhancedSignal(
        direction=direction,
        level=signal_level,
        confidence=confidence,
        long_strength=mtf_signal['long_strength'],
        short_strength=mtf_signal['short_strength'],
        trend_filter_passed=trend_filter_passed,
        sentiment_score=sentiment.sentiment_score,
        model_consistency=ensemble_pred.consistency,
        is_abnormal_market=sentiment.is_abnormal,
        suggested_position_ratio=suggested_position_ratio,
    )


def generate_enhanced_signals_for_backtest(
    df_15m: pd.DataFrame,
    ensemble_model: EnsembleModel,
    cfg: Config,
) -> pd.DataFrame:
    """批量生成回测信号（简化版，不包含实时API数据）。
    
    Args:
        df_15m: 15分钟K线数据
        ensemble_model: 集成模型
        cfg: 配置对象
    
    Returns:
        信号DataFrame，包含时间、方向、置信度等列
    """
    # 1. 构建多时间周期特征（用于趋势分析）
    mtf_features = build_multi_timeframe_features(df_15m, cfg)
    
    # 2. 构建基础特征（用于模型预测，保证和训练时一致）
    from .features import build_features_and_labels
    feature_label_data = build_features_and_labels(cfg, df_15m)
    base_features = feature_label_data.features
    
    # 3. 对齐两个特征集
    min_len = min(len(base_features), len(mtf_features.features))
    base_features = base_features.iloc[:min_len].copy()
    mtf_features_aligned = mtf_features.features.iloc[:min_len].copy()
    mtf_times_aligned = mtf_features.times.iloc[:min_len].copy()
    
    # ⚡ 优化：批量预测而不是逐行预测
    print(f"开始批量预测 {len(base_features)} 条数据...")
    ensemble_pred = predict_ensemble(ensemble_model, base_features)
    
    # 4. 批量生成初始信号（模型预测）
    strategy_cfg = cfg.strategy
    min_prob = float(strategy_cfg.get("min_prob", 0.35))
    prob_gap = float(strategy_cfg.get("prob_gap_long", 0.1))
    
    # 向量化信号判断
    prob_long = ensemble_pred.prob_long
    prob_short = ensemble_pred.prob_short
    
    initial_direction = np.zeros(len(prob_long), dtype=int)
    long_mask = (prob_long > min_prob) & ((prob_long - prob_short) > prob_gap)
    short_mask = (prob_short > min_prob) & ((prob_short - prob_long) > prob_gap)
    initial_direction[long_mask] = 1
    initial_direction[short_mask] = -1
    
    # === 方案B：入场确认机制 + 成交量确认 ===
    # 模型信号 + 价格确认（突破K线高点或收阳/阴）+ 成交量确认
    enable_entry_confirmation = strategy_cfg.get("enable_entry_confirmation", True)
    require_volume_confirmation = strategy_cfg.get("require_volume_confirmation", False)
    volume_threshold = float(strategy_cfg.get("volume_threshold", 1.2))
    
    confirmed_direction = initial_direction.copy()
    
    if enable_entry_confirmation and min_len > 1:
        print("应用入场确认机制...")
        
        # 从 df_15m 中提取价格数据（对齐长度）
        df_aligned = df_15m.iloc[-min_len:].reset_index(drop=True)
        
        # 下一根K线的价格数据
        next_high = df_aligned["high"].shift(-1)
        next_low = df_aligned["low"].shift(-1)
        next_close = df_aligned["close"].shift(-1)
        next_open = df_aligned["open"].shift(-1)
        current_high = df_aligned["high"]
        current_low = df_aligned["low"]
        
        # 做多确认：下一根K线突破当前K线高点 或 收阳
        long_breakthrough = next_high > current_high
        long_bullish_candle = next_close > next_open
        long_confirmed = long_breakthrough | long_bullish_candle
        
        # 做空确认：下一根K线跌破当前K线低点 或 收阴
        short_breakthrough = next_low < current_low
        short_bearish_candle = next_close < next_open
        short_confirmed = short_breakthrough | short_bearish_candle
        
        # DeepSeek优化：成交量确认
        if require_volume_confirmation:
            volume_ma20 = df_aligned["volume"].rolling(20).mean()
            volume_spike = df_aligned["volume"] > (volume_ma20 * volume_threshold)
            long_confirmed = long_confirmed & volume_spike
            short_confirmed = short_confirmed & volume_spike
        
        # 应用确认逻辑
        confirmed_direction[(initial_direction == 1) & ~long_confirmed] = 0  # 做多未确认→观望
        confirmed_direction[(initial_direction == -1) & ~short_confirmed] = 0  # 做空未确认→观望
        
        # 最后一根K线无法确认（没有下一根）
        confirmed_direction[-1] = 0
        
        n_long_initial = (initial_direction == 1).sum()
        n_long_confirmed = (confirmed_direction == 1).sum()
        n_short_initial = (initial_direction == -1).sum()
        n_short_confirmed = (confirmed_direction == -1).sum()
        
        print(f"确认结果 - 做多: {n_long_initial} → {n_long_confirmed}, 做空: {n_short_initial} → {n_short_confirmed}")
    
    # 简化置信度计算（使用模型一致性作为主要依据）
    confidence = ensemble_pred.consistency
    position_ratio = confidence * 0.8  # 基础仓位
    position_ratio[confirmed_direction == 0] = 0.0  # 无信号时仓位为0
    
    # 构建结果DataFrame
    signals_df = pd.DataFrame({
        'direction': confirmed_direction,
        'confidence': confidence,
        'position_ratio': position_ratio,
        'model_consistency': ensemble_pred.consistency,
    })
    signals_df.index = mtf_times_aligned
    signals_df.index.name = 'time'
    
    # 统计
    n_long = (confirmed_direction == 1).sum()
    n_short = (confirmed_direction == -1).sum() 
    n_flat = (confirmed_direction == 0).sum()
    print(f"最终信号统计 - 做多: {n_long}, 做空: {n_short}, 观望: {n_flat}")
    
    return signals_df
