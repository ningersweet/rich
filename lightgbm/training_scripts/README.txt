===============================================================================
训练脚本目录说明
===============================================================================

【目录用途】
存放所有模型训练脚本，用于训练LightGBM两阶段风险收益模型。

【脚本列表】
- train_2024_model.py   : 2024年模型训练脚本（2018-2024年数据）

【使用方法】
cd /Users/lemonshwang/project/rich/lightgbm
python training_scripts/train_2024_model.py

【训练输出】
模型保存位置: lightgbm/models/final_2024_dynamic/
包含文件:
- risk_reward_model.txt
- direction_model.txt
- period_model.txt
- top30_features.txt
- training_info.txt

【注意事项】
⚠️ 训练数据与回测数据必须严格分离
⚠️ 训练完成后才能运行回测脚本
===============================================================================
