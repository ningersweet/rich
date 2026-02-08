#!/usr/bin/env python3
"""
盈亏比驱动标签构建模块

基于DeepSeek建议的核心逻辑：
1. 动态目标收益（max(1%, 1倍ATR)）
2. 计算未来实际最大盈利和最大亏损
3. 构建三个标签：方向、持有周期、盈亏比
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class RiskRewardLabelBuilder:
    """
    盈亏比标签构建器
    
    核心目标：构建高质量标签，用于预测交易的盈亏比
    """
    
    def __init__(
        self,
        target_return: float = 0.01,  # 基础目标收益率1%
        max_holding_period: int = 50,  # 最大持有50根K线（12.5小时）
        min_rr_ratio: float = 1.5,     # 最小盈亏比阈值
        volatility_factor: float = 1.0, # ATR调整因子
    ):
        """
        初始化标签构建器
        
        Args:
            target_return: 基础目标收益率
            max_holding_period: 观察未来多少根K线
            min_rr_ratio: 最小盈亏比，低于此值标记为观望
            volatility_factor: 波动率调整因子（动态目标 = ATR * factor）
        """
        self.target_return = target_return
        self.max_holding_period = max_holding_period
        self.min_rr_ratio = min_rr_ratio
        self.volatility_factor = volatility_factor
        
        logger.info(f"标签构建器初始化: 目标收益={target_return*100:.1f}%, "
                   f"最大周期={max_holding_period}, 最小盈亏比={min_rr_ratio}")
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR（平均真实波幅）"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def build_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建多任务标签
        
        Args:
            df: 包含OHLCV的DataFrame（原始数据，必须包含open/high/low/close/volume）
        
        Returns:
            标签DataFrame，包含：
            - direction: 方向 (1=做多, 0=观望, -1=做空)
            - holding_period: 最优持有周期（K线数）
            - risk_reward: 盈亏比
            - expected_profit_pct: 预期盈利百分比
            - expected_loss_pct: 预期亏损百分比
        """
        df = df.copy()
        n_samples = len(df)
        
        # 计算ATR（如果没有）
        if 'atr_14' not in df.columns:
            logger.info("计算ATR...")
            atr = self._calculate_atr(df, period=14)
            df['atr_14'] = atr
        
        # 提取价格数组（numpy更快）
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        atr = df['atr_14'].values
        
        # 初始化标签数组
        direction_labels = np.zeros(n_samples, dtype=np.int8)
        holding_labels = np.zeros(n_samples, dtype=np.int16)
        rr_labels = np.zeros(n_samples, dtype=np.float32)
        expected_profit = np.zeros(n_samples, dtype=np.float32)
        expected_loss = np.zeros(n_samples, dtype=np.float32)
        
        # 为每个时间点计算标签
        logger.info(f"开始构建标签，总样本数: {n_samples}")
        
        for i in range(n_samples - self.max_holding_period):
            if i % 10000 == 0:
                logger.debug(f"处理进度: {i}/{n_samples}")
            
            current_price = close[i]
            current_atr = atr[i]
            
            # 跳过无效数据
            if np.isnan(current_price) or np.isnan(current_atr) or current_atr <= 0:
                continue
            
            # 计算动态目标收益：max(基础目标, ATR比例)
            atr_pct = current_atr / current_price
            dynamic_target = max(self.target_return, atr_pct * self.volatility_factor)
            
            # 获取未来价格序列
            future_slice = slice(i+1, i+1+self.max_holding_period)
            future_highs = high[future_slice]
            future_lows = low[future_slice]
            future_closes = close[future_slice]
            
            # 计算多头机会
            long_result = self._calculate_long_opportunity(
                current_price, future_highs, future_lows, future_closes, dynamic_target
            )
            
            # 计算空头机会
            short_result = self._calculate_short_opportunity(
                current_price, future_highs, future_lows, future_closes, dynamic_target
            )
            
            # 决策逻辑：选择盈亏比更高的方向
            long_rr = long_result['rr']
            short_rr = short_result['rr']
            
            if long_rr >= self.min_rr_ratio and short_rr >= self.min_rr_ratio:
                # 双向都有机会，选择盈亏比更高的
                if long_rr >= short_rr:
                    direction_labels[i] = 1
                    holding_labels[i] = long_result['period']
                    rr_labels[i] = long_rr
                    expected_profit[i] = long_result['profit_pct']
                    expected_loss[i] = long_result['loss_pct']
                else:
                    direction_labels[i] = -1
                    holding_labels[i] = short_result['period']
                    rr_labels[i] = short_rr
                    expected_profit[i] = short_result['profit_pct']
                    expected_loss[i] = short_result['loss_pct']
            
            elif long_rr >= self.min_rr_ratio:
                direction_labels[i] = 1
                holding_labels[i] = long_result['period']
                rr_labels[i] = long_rr
                expected_profit[i] = long_result['profit_pct']
                expected_loss[i] = long_result['loss_pct']
            
            elif short_rr >= self.min_rr_ratio:
                direction_labels[i] = -1
                holding_labels[i] = short_result['period']
                rr_labels[i] = short_rr
                expected_profit[i] = short_result['profit_pct']
                expected_loss[i] = short_result['loss_pct']
            
            else:
                # 没有足够盈亏比的机会，观望
                direction_labels[i] = 0
                holding_labels[i] = 0
                rr_labels[i] = 0.0
                expected_profit[i] = 0.0
                expected_loss[i] = 0.0
        
        # 创建标签DataFrame
        labels_df = pd.DataFrame({
            'direction': direction_labels,
            'holding_period': holding_labels,
            'risk_reward': rr_labels,
            'expected_profit_pct': expected_profit,
            'expected_loss_pct': expected_loss,
        }, index=df.index)
        
        # 分析标签分布
        self._analyze_labels(labels_df)
        
        return labels_df
    
    def _calculate_long_opportunity(
        self,
        current_price: float,
        future_highs: np.ndarray,
        future_lows: np.ndarray,
        future_closes: np.ndarray,
        target: float
    ) -> Dict:
        """
        计算多头机会
        
        Returns:
            dict: {
                'rr': 盈亏比,
                'period': 持有周期,
                'profit_pct': 预期盈利百分比,
                'loss_pct': 预期亏损百分比
            }
        """
        # 计算未来收益率序列
        future_returns = (future_closes - current_price) / current_price
        
        # 检查是否能达到目标收益
        reached = future_returns >= target
        
        if not reached.any():
            return {'rr': 0.0, 'period': 0, 'profit_pct': 0.0, 'loss_pct': 0.0}
        
        # 找到第一次达到目标的位置
        first_reach = np.argmax(reached)
        best_period = first_reach + 1
        
        # 计算期间的最大盈利和最大亏损
        period_highs = future_highs[:best_period]
        period_lows = future_lows[:best_period]
        
        max_profit_pct = (period_highs.max() - current_price) / current_price
        max_loss_pct = abs((period_lows.min() - current_price) / current_price)
        
        # 避免除零
        if max_loss_pct < 0.001:
            max_loss_pct = 0.001
        
        # 计算盈亏比
        rr = max_profit_pct / max_loss_pct
        
        return {
            'rr': rr,
            'period': best_period,
            'profit_pct': max_profit_pct,
            'loss_pct': max_loss_pct
        }
    
    def _calculate_short_opportunity(
        self,
        current_price: float,
        future_highs: np.ndarray,
        future_lows: np.ndarray,
        future_closes: np.ndarray,
        target: float
    ) -> Dict:
        """
        计算空头机会
        
        Returns:
            dict: 同上
        """
        # 计算未来收益率序列（做空）
        future_returns = (current_price - future_closes) / current_price
        
        # 检查是否能达到目标收益
        reached = future_returns >= target
        
        if not reached.any():
            return {'rr': 0.0, 'period': 0, 'profit_pct': 0.0, 'loss_pct': 0.0}
        
        # 找到第一次达到目标的位置
        first_reach = np.argmax(reached)
        best_period = first_reach + 1
        
        # 计算期间的最大盈利和最大亏损
        period_highs = future_highs[:best_period]
        period_lows = future_lows[:best_period]
        
        # 做空：价格下跌是盈利，价格上涨是亏损
        max_profit_pct = (current_price - period_lows.min()) / current_price
        max_loss_pct = abs((period_highs.max() - current_price) / current_price)
        
        # 避免除零
        if max_loss_pct < 0.001:
            max_loss_pct = 0.001
        
        # 计算盈亏比
        rr = max_profit_pct / max_loss_pct
        
        return {
            'rr': rr,
            'period': best_period,
            'profit_pct': max_profit_pct,
            'loss_pct': max_loss_pct
        }
    
    def _analyze_labels(self, labels_df: pd.DataFrame):
        """分析标签分布"""
        logger.info("=" * 60)
        logger.info("标签分布分析")
        logger.info("=" * 60)
        
        total = len(labels_df)
        logger.info(f"总样本数: {total}")
        
        # 方向分布
        direction_counts = labels_df['direction'].value_counts()
        logger.info("\n方向分布:")
        for direction, label in [(1, '做多'), (0, '观望'), (-1, '做空')]:
            count = direction_counts.get(direction, 0)
            pct = count / total * 100
            logger.info(f"  {label}: {count:,} ({pct:.1f}%)")
        
        # 交易样本统计
        active_trades = labels_df[labels_df['direction'] != 0]
        if len(active_trades) > 0:
            logger.info(f"\n交易样本统计 (n={len(active_trades):,}):")
            
            logger.info(f"  持有周期 - 均值: {active_trades['holding_period'].mean():.1f}, "
                       f"中位数: {active_trades['holding_period'].median():.1f}")
            
            logger.info(f"  盈亏比 - 均值: {active_trades['risk_reward'].mean():.2f}, "
                       f"中位数: {active_trades['risk_reward'].median():.2f}")
            
            logger.info(f"  盈亏比>2.0: {(active_trades['risk_reward'] > 2.0).sum():,} "
                       f"({(active_trades['risk_reward'] > 2.0).sum()/len(active_trades)*100:.1f}%)")
            
            logger.info(f"  盈亏比>3.0: {(active_trades['risk_reward'] > 3.0).sum():,} "
                       f"({(active_trades['risk_reward'] > 3.0).sum()/len(active_trades)*100:.1f}%)")
        
        logger.info("=" * 60)
    
    def get_label_statistics(self, labels_df: pd.DataFrame) -> Dict:
        """
        获取标签统计信息（用于验证）
        
        Returns:
            dict: 统计信息
        """
        active_trades = labels_df[labels_df['direction'] != 0]
        
        stats = {
            'total_samples': len(labels_df),
            'long_samples': (labels_df['direction'] == 1).sum(),
            'short_samples': (labels_df['direction'] == -1).sum(),
            'flat_samples': (labels_df['direction'] == 0).sum(),
            'active_ratio': len(active_trades) / len(labels_df) if len(labels_df) > 0 else 0,
        }
        
        if len(active_trades) > 0:
            stats.update({
                'avg_holding_period': active_trades['holding_period'].mean(),
                'median_holding_period': active_trades['holding_period'].median(),
                'avg_risk_reward': active_trades['risk_reward'].mean(),
                'median_risk_reward': active_trades['risk_reward'].median(),
                'high_rr_ratio': (active_trades['risk_reward'] > 2.0).sum() / len(active_trades),
            })
        
        return stats


def main():
    """测试标签构建"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from btc_quant.data import load_klines
    from btc_quant.features import build_features_and_labels
    from btc_quant.config import load_config
    from pathlib import Path
    
    # 加载配置
    cfg = load_config(Path('/Users/lemonshwang/project/rich/config.yaml'))
    
    # 加载数据
    print("加载K线数据...")
    klines = load_klines(cfg)
    print(f"加载完成: {len(klines)} 根K线")
    
    # 构建旧标签（用于对比）
    print("\n构建突破标签（用于对比）...")
    feature_label_data = build_features_and_labels(cfg, klines)
    labels_old = feature_label_data.labels
    print(f"旧标签构建完成")
    
    # 构建新标签
    print("\n=" * 60)
    print("构建盈亏比标签")
    print("=" * 60)
    label_builder = RiskRewardLabelBuilder(
        target_return=0.01,
        max_holding_period=50,
        min_rr_ratio=1.5,
        volatility_factor=1.0
    )
    
    labels_new = label_builder.build_labels(klines)  # 直接使用原始klines
    
    # 对比旧标签
    print("\n" + "=" * 60)
    print("对比分析")
    print("=" * 60)
    print("\n旧标签分布（突破标签）:")
    old_counts = labels_old.value_counts()
    for label in [1, 0, -1]:
        count = old_counts.get(label, 0)
        print(f"  {['做空', '观望', '做多'][label+1]}: {count:,} ({count/len(labels_old)*100:.1f}%)")
    
    print("\n新标签方向分布（盈亏比标签）:")
    new_counts = labels_new['direction'].value_counts()
    for label in [1, 0, -1]:
        count = new_counts.get(label, 0)
        print(f"  {['做空', '观望', '做多'][label+1]}: {count:,} ({count/len(labels_new)*100:.1f}%)")
    
    # 获取统计信息
    stats = label_builder.get_label_statistics(labels_new)
    print("\n盈亏比标签关键统计:")
    print(f"  高质量交易（RR>2.0）占比: {stats.get('high_rr_ratio', 0)*100:.1f}%")
    print(f"  平均盈亏比: {stats.get('avg_risk_reward', 0):.2f}")
    print(f"  平均持有周期: {stats.get('avg_holding_period', 0):.1f} 根K线")
    
    # 保存样本
    print("\n样本数据（盈亏比>2.5的前10个）:")
    high_rr_samples = labels_new[labels_new['risk_reward'] > 2.5].head(10)
    sample = pd.concat([
        klines.loc[high_rr_samples.index, ['close', 'high', 'low']],
        high_rr_samples
    ], axis=1)
    pd.set_option('display.width', 150)
    pd.set_option('display.max_columns', 10)
    print(sample.to_string())
    
    print("\n" + "=" * 60)
    print("✅ 标签构建测试完成！")
    print("=" * 60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()
