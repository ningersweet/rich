===============================================================================
回测脚本目录说明
===============================================================================

【目录用途】
存放所有模型回测脚本，用于在样本外数据上验证模型性能。

【脚本列表】
- backtest_2024_model.py  : 回测入口脚本
- backtest_engine.py      : 回测引擎（核心逻辑）

【使用方法】
cd /Users/lemonshwang/project/rich/lightgbm
python backtest_scripts/backtest_2024_model.py

【回测输出】
交易记录保存: lightgbm/backtest_results/final_2024_dynamic_YYYYMMDD_HHMMSS.csv

【注意事项】
⚠️ 必须先训练模型（training_scripts/train_2024_model.py）
⚠️ 回测使用2025-2026年样本外数据
⚠️ 回测参数应与实盘保持一致
===============================================================================
