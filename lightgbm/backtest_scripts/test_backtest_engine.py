#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测引擎单元测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from backtest_scripts.backtest_engine import calculate_dynamic_exposure


def test_calculate_dynamic_exposure_basic():
    """测试基础动态敞口计算"""
    # 测试正常情况
    exposure = calculate_dynamic_exposure(predicted_rr=2.5, direction_prob=0.7)
    assert 1.0 <= exposure <= 10.0, f"敞口应在1-10范围内，实际为{exposure}"
    
    # 测试高盈亏比和高概率
    exposure = calculate_dynamic_exposure(predicted_rr=5.0, direction_prob=0.9)
    assert exposure >= 1.0, f"高信号质量应产生较高敞口，实际为{exposure}"
    
    # 测试低信号质量
    exposure = calculate_dynamic_exposure(predicted_rr=1.0, direction_prob=0.5)
    assert exposure >= 1.0, f"低信号质量敞口应至少为1，实际为{exposure}"


def test_calculate_dynamic_exposure_drawdown_penalty():
    """测试回撤惩罚"""
    # 无回撤情况
    exposure_no_dd = calculate_dynamic_exposure(predicted_rr=2.5, direction_prob=0.7, current_drawdown=0)
    
    # 有回撤情况
    exposure_with_dd = calculate_dynamic_exposure(predicted_rr=2.5, direction_prob=0.7, current_drawdown=0.1)
    
    # 回撤应降低敞口
    assert exposure_with_dd <= exposure_no_dd, f"回撤应降低敞口，无回撤:{exposure_no_dd}, 有回撤:{exposure_with_dd}"


def test_calculate_dynamic_exposure_consecutive_losses():
    """测试连续亏损惩罚"""
    # 无连续亏损
    exposure_no_loss = calculate_dynamic_exposure(predicted_rr=2.5, direction_prob=0.7, consecutive_losses=0)
    
    # 有连续亏损
    exposure_with_loss = calculate_dynamic_exposure(predicted_rr=2.5, direction_prob=0.7, consecutive_losses=3)
    
    # 连续亏损应降低敞口
    assert exposure_with_loss <= exposure_no_loss, f"连续亏损应降低敞口，无亏损:{exposure_no_loss}, 有亏损:{exposure_with_loss}"


def test_calculate_dynamic_exposure_boundaries():
    """测试边界情况"""
    # 测试最小敞口
    exposure = calculate_dynamic_exposure(predicted_rr=1.0, direction_prob=0.5, current_drawdown=0.5, consecutive_losses=10)
    assert exposure >= 1.0, f"即使条件很差，敞口也应至少为1，实际为{exposure}"
    
    # 测试最大敞口
    exposure = calculate_dynamic_exposure(predicted_rr=10.0, direction_prob=1.0, current_drawdown=0, consecutive_losses=0, max_exposure=10.0)
    assert exposure <= 10.0, f"敞口不应超过最大值，实际为{exposure}"


def test_calculate_dynamic_exposure_max_exposure_parameter():
    """测试max_exposure参数"""
    # 测试不同的最大敞口限制
    exposure = calculate_dynamic_exposure(predicted_rr=5.0, direction_prob=0.9, max_exposure=5.0)
    assert exposure <= 5.0, f"敞口应不超过指定的最大值5，实际为{exposure}"
    
    exposure = calculate_dynamic_exposure(predicted_rr=5.0, direction_prob=0.9, max_exposure=20.0)
    assert exposure <= 20.0, f"敞口应不超过指定的最大值20，实际为{exposure}"


if __name__ == '__main__':
    # 运行所有测试
    test_calculate_dynamic_exposure_basic()
    test_calculate_dynamic_exposure_drawdown_penalty()
    test_calculate_dynamic_exposure_consecutive_losses()
    test_calculate_dynamic_exposure_boundaries()
    test_calculate_dynamic_exposure_max_exposure_parameter()
    print("✅ 所有测试通过！")