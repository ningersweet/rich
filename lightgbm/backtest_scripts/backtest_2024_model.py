#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
LightGBM 两阶段模型回测脚本 - 复利+动态敞口策略
===============================================================================

【功能说明】
使用训练好的2024年模型在2025-2026年样本外数据上进行回测验证。

【回测配置】
- 模型路径: models/final_2024_dynamic/
- 回测数据: 2025-01-01 至 2026-02-20（样本外）
- 资金模式: 复利（权益滚动增长）
- 敞口策略: 动态敞口 1-10倍

【使用方法】
cd /Users/lemonshwang/project/rich/lightgbm
python backtest_scripts/backtest_2024_model.py

【作者】Qoder AI
【日期】2026-02-21
===============================================================================
"""
# 导入回测引擎和主函数
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_scripts.backtest_engine import run_backtest

if __name__ == '__main__':
    run_backtest()
