"""
动态止损模块单元测试
"""

import pytest
import numpy as np
import pandas as pd
from btc_quant.dynamic_stop_loss import (
    calculate_atr_stop_loss,
    calculate_adaptive_atr_multiplier,
    TrailingStopManager,
    calculate_dynamic_stop_loss_params
)


class TestATRStopLoss:
    """测试ATR动态止损功能"""
    
    def test_long_position_stop_loss(self):
        """测试做多的ATR止损计算"""
        entry_price = 100000.0
        atr = 1500.0
        direction = 1  # 做多
        k = 2.0
        
        stop_price, stop_pct = calculate_atr_stop_loss(entry_price, atr, direction, k)
        
        # 止损价 = 100000 - 2*1500 = 97000
        assert abs(stop_price - 97000.0) < 1.0
        # 止损百分比 = -3%
        assert abs(stop_pct - (-0.03)) < 0.001
    
    def test_short_position_stop_loss(self):
        """测试做空的ATR止损计算"""
        entry_price = 100000.0
        atr = 1500.0
        direction = -1  # 做空
        k = 2.0
        
        stop_price, stop_pct = calculate_atr_stop_loss(entry_price, atr, direction, k)
        
        # 止损价 = 100000 + 2*1500 = 103000
        assert abs(stop_price - 103000.0) < 1.0
        # 止损百分比 = -3% (对于做空)
        assert abs(stop_pct - (-0.03)) < 0.001
    
    def test_stop_loss_limits(self):
        """测试止损限制（防止过度激进或保守）"""
        entry_price = 100000.0
        
        # 测试过小的ATR（会被限制到最小1%）
        small_atr = 100.0
        k = 2.0
        stop_price, stop_pct = calculate_atr_stop_loss(
            entry_price, small_atr, 1, k,
            min_stop_loss_pct=0.01, max_stop_loss_pct=0.05
        )
        assert stop_pct >= -0.05  # 不超过最大止损
        assert stop_pct <= -0.01  # 不低于最小止损
        
        # 测试过大的ATR（会被限制到最大5%）
        large_atr = 5000.0
        stop_price, stop_pct = calculate_atr_stop_loss(
            entry_price, large_atr, 1, k,
            min_stop_loss_pct=0.01, max_stop_loss_pct=0.05
        )
        assert stop_pct >= -0.05
        assert stop_pct <= -0.01


class TestAdaptiveATRMultiplier:
    """测试自适应ATR倍数"""
    
    def test_high_volatility_adjustment(self):
        """高波动期应放宽止损"""
        k = calculate_adaptive_atr_multiplier(
            volatility_percentile=0.9,
            trend_strength=0.0,
            base_k=2.0
        )
        assert k > 2.0  # 应该放宽
        assert k <= 3.5  # 不超过上限
    
    def test_low_volatility_adjustment(self):
        """低波动期应收紧止损"""
        k = calculate_adaptive_atr_multiplier(
            volatility_percentile=0.2,
            trend_strength=0.0,
            base_k=2.0
        )
        assert k < 2.0  # 应该收紧
        assert k >= 1.5  # 不低于下限
    
    def test_strong_trend_adjustment(self):
        """强趋势期应略放宽止损"""
        k = calculate_adaptive_atr_multiplier(
            volatility_percentile=0.5,
            trend_strength=0.8,
            base_k=2.0
        )
        assert k > 2.0  # 应该放宽


class TestTrailingStopManager:
    """测试移动止盈管理器"""
    
    def test_initialization(self):
        """测试初始化"""
        manager = TrailingStopManager(trailing_pct=0.5, min_profit_pct=0.01)
        assert not manager.is_active
        assert manager.trailing_stop_price is None
    
    def test_long_trailing_stop(self):
        """测试做多的移动止盈"""
        manager = TrailingStopManager(trailing_pct=0.5, min_profit_pct=0.01)
        entry_price = 100000.0
        direction = 1
        
        # 价格未达到最小利润，不激活
        result = manager.update(100500.0, entry_price, direction)
        assert result is None
        assert not manager.is_active
        
        # 价格上涨到+2%，激活trailing stop
        result = manager.update(102000.0, entry_price, direction)
        assert result is not None
        assert manager.is_active
        assert manager.highest_price == 102000.0
        
        # 价格继续上涨到+5%
        result = manager.update(105000.0, entry_price, direction)
        assert manager.highest_price == 105000.0
        
        # 价格回落，但未触发trailing stop
        assert not manager.should_exit(103000.0, direction)
        
        # 价格继续回落，触发trailing stop（利润回撤50%）
        # 最大利润5%，允许回撤50%，即保留2.5%
        # trailing_stop = 100000 * 1.025 = 102500
        assert manager.should_exit(102400.0, direction)
    
    def test_short_trailing_stop(self):
        """测试做空的移动止盈"""
        manager = TrailingStopManager(trailing_pct=0.5, min_profit_pct=0.01)
        entry_price = 100000.0
        direction = -1
        
        # 价格下跌到-2%，激活trailing stop
        result = manager.update(98000.0, entry_price, direction)
        assert result is not None
        assert manager.is_active
        assert manager.lowest_price == 98000.0
        
        # 价格继续下跌到-5%
        result = manager.update(95000.0, entry_price, direction)
        assert manager.lowest_price == 95000.0
        
        # 价格反弹，触发trailing stop
        assert manager.should_exit(97600.0, direction)
    
    def test_reset(self):
        """测试重置功能"""
        manager = TrailingStopManager()
        manager.update(102000.0, 100000.0, 1)
        assert manager.is_active
        
        manager.reset()
        assert not manager.is_active
        assert manager.highest_price is None


class TestDynamicStopLossParams:
    """测试动态止损参数计算"""
    
    def test_calculate_params(self):
        """测试参数计算"""
        # 创建模拟K线数据
        n = 150
        klines = pd.DataFrame({
            'close': 100000 + np.random.randn(n) * 1000,
            'high': 100500 + np.random.randn(n) * 1000,
            'low': 99500 + np.random.randn(n) * 1000,
            'atr_14': 1500 + np.random.randn(n) * 200
        })
        
        params = calculate_dynamic_stop_loss_params(klines, lookback_window=100)
        
        assert 'atr' in params
        assert 'volatility_percentile' in params
        assert 'trend_strength' in params
        assert 'recommended_k' in params
        
        # 验证范围
        assert 0 <= params['volatility_percentile'] <= 1
        assert -1 <= params['trend_strength'] <= 1
        assert 1.5 <= params['recommended_k'] <= 3.5
    
    def test_missing_atr_column(self):
        """测试缺少ATR列时的错误处理"""
        klines = pd.DataFrame({
            'close': [100000, 101000, 102000]
        })
        
        with pytest.raises(ValueError, match="必须包含atr_14列"):
            calculate_dynamic_stop_loss_params(klines)


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v'])
