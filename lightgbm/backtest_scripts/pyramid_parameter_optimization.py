#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
金字塔加仓策略参数优化脚本
===============================================================================

【功能说明】
通过网格搜索方法优化金字塔加仓策略的参数组合，寻找最佳参数配置。

【优化参数】
1. pyramid_profit_threshold: 盈利阈值（允许加仓的最小盈利百分比）
2. pyramid_min_rr: 加仓信号最小盈亏比阈值
3. pyramid_min_prob: 加仓信号最小概率阈值
4. pyramid_max_count: 最大加仓次数
5. pyramid_min_bars: 距上次加仓最小K线数
6. max_total_exposure: 总敞口上限（含加仓）

【优化目标】
- 最大化总收益率
- 最大化收益风险比（总收益率/最大回撤）
- 在收益和风险之间寻找平衡

【使用方法】
cd /Users/lemonshwang/project/rich/lightgbm
python backtest_scripts/pyramid_parameter_optimization.py

【作者】Qoder AI
【日期】2026-02-21
===============================================================================
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
import itertools
import json

# 导入回测引擎和模型
from btc_quant.config import load_config
from btc_quant.data import load_klines
from btc_quant.features import build_features_and_labels
from btc_quant.risk_reward_model import TwoStageRiskRewardStrategy
from backtest_scripts.backtest_engine_pyramid_shared import pyramid_backtest_with_compounding

# 日志配置
log_file = Path('logs') / f'pyramid_parameter_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"日志文件: {log_file.absolute()}")

# 回测配置（固定）
BACKTEST_START = '2025-01-01T00:00:00Z'
BACKTEST_END = '2026-02-20T23:59:59Z'
MODEL_DIR = Path('models/final_2024_dynamic')
INITIAL_BALANCE = 1000.0
MAX_EXPOSURE = 10.0
STOP_LOSS_PCT = -0.03
MAX_DAILY_LOSS_PCT = -0.20
MAX_DRAWDOWN_PAUSE = 0.10
USE_TRAILING_STOP = True
MAX_HOLDING_PERIOD = 20
RR_THRESHOLD = 1.0
PROB_THRESHOLD = 0.0

# 金字塔参数搜索空间（测试模式：每个参数2个值，共64种组合）
PARAM_SEARCH_SPACE = {
    'pyramid_profit_threshold': [0.01, 0.02],            # 1%, 2%
    'pyramid_min_rr': [3.0, 3.5],                        # 3.0, 3.5
    'pyramid_min_prob': [0.75, 0.8],                     # 75%, 80%
    'pyramid_max_count': [3, 4],                         # 3次, 4次
    'pyramid_min_bars': [5, 7],                          # 5根, 7根K线
    'max_total_exposure': [15.0, 18.0]                   # 15倍, 18倍
}

# 当前最佳参数配置（来自config.yaml）
DEFAULT_PARAMS = {
    'pyramid_enabled': True,
    'pyramid_profit_threshold': 0.01,
    'pyramid_min_rr': 3.0,
    'pyramid_min_prob': 0.75,
    'pyramid_max_count': 3,
    'pyramid_min_bars': 5,
    'max_total_exposure': 15.0
}

def load_data_and_predictions() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载K线数据和生成预测信号（单次加载，多次使用）
    
    返回:
        klines_backtest: 回测K线数据
        predictions: 预测信号DataFrame
    """
    logger.info("=" * 80)
    logger.info("📊 加载数据和生成预测信号")
    logger.info("=" * 80)
    
    # 加载配置
    cfg = load_config()
    
    logger.info(f"\n回测时间范围:{BACKTEST_START}至{BACKTEST_END}")
    logger.info("加载K线数据...")
    klines_all = load_klines(cfg)
    klines_all['close_time'] = pd.to_datetime(klines_all['close_time'])
    backtest_start_ts, backtest_end_ts = pd.Timestamp(BACKTEST_START), pd.Timestamp(BACKTEST_END)
    klines_backtest = klines_all[(klines_all['close_time'] >= backtest_start_ts) & 
                                  (klines_all['close_time'] <= backtest_end_ts)].reset_index(drop=True)
    logger.info(f"回测集K线数量:{len(klines_backtest)}")
    
    logger.info("构建特征...")
    feature_label_data = build_features_and_labels(cfg, klines_backtest)
    X_backtest_full = feature_label_data.features.reset_index(drop=True)
    min_len = min(len(X_backtest_full), len(klines_backtest))
    X_backtest_full = X_backtest_full.iloc[:min_len]
    klines_backtest = klines_backtest.iloc[:min_len].reset_index(drop=True)
    logger.info(f"对齐后样本数:{len(X_backtest_full)}")
    
    logger.info(f"加载模型:{MODEL_DIR}")
    strategy = TwoStageRiskRewardStrategy()
    strategy.load(MODEL_DIR)
    with open(MODEL_DIR / 'top30_features.txt', 'r') as f:
        top_30_features = [line.strip() for line in f.readlines()]
    logger.info(f"特征数量:{len(top_30_features)}")
    X_backtest_top30 = X_backtest_full[top_30_features]
    
    logger.info("生成预测信号...")
    predictions_dict = strategy.predict(X_backtest_top30, rr_threshold=RR_THRESHOLD, prob_threshold=PROB_THRESHOLD)
    predictions = pd.DataFrame({
        'predicted_rr': predictions_dict['predicted_rr'],
        'direction': predictions_dict['direction'],
        'holding_period': predictions_dict['holding_period'].clip(1, MAX_HOLDING_PERIOD),
        'direction_prob': predictions_dict['direction_prob'],
        'should_trade': predictions_dict['should_trade']
    })
    logger.info(f"总样本数:{len(predictions)}")
    logger.info(f"应交易样本:{predictions['should_trade'].sum()}")
    logger.info(f"交易比例:{predictions['should_trade'].sum()/len(predictions)*100:.2f}%")
    
    logger.info(f"✅ 数据准备完成: K线{len(klines_backtest)}条, 预测{len(predictions)}条")
    
    return klines_backtest, predictions

def run_single_backtest(
    klines: pd.DataFrame,
    predictions: pd.DataFrame,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    使用指定参数运行单次回测
    
    参数:
        klines: K线数据
        predictions: 预测信号
        params: 金字塔参数字典
        
    返回:
        回测结果字典
    """
    try:
        result = pyramid_backtest_with_compounding(
            klines=klines,
            predictions=predictions,
            initial_balance=INITIAL_BALANCE,
            max_total_exposure=params['max_total_exposure'],
            stop_loss_pct=STOP_LOSS_PCT,
            max_daily_loss_pct=MAX_DAILY_LOSS_PCT,
            max_drawdown_pause=MAX_DRAWDOWN_PAUSE,
            use_trailing_stop=USE_TRAILING_STOP,
            pyramid_enabled=params['pyramid_enabled'],
            pyramid_profit_threshold=params['pyramid_profit_threshold'],
            pyramid_min_rr=params['pyramid_min_rr'],
            pyramid_min_prob=params['pyramid_min_prob'],
            pyramid_max_count=params['pyramid_max_count'],
            pyramid_min_bars=params['pyramid_min_bars']
        )
        
        # 计算收益风险比
        risk_return_ratio = result['total_return'] / max(result['max_drawdown'], 0.01)
        
        # 计算综合评分（权重：总收益率60%，收益风险比30%，胜率10%）
        # 归一化处理（相对于最优值）
        total_return_score = min(result['total_return'] / 1e12, 1.0)  # 假设最大1万亿%
        risk_return_score = min(risk_return_ratio / 1e10, 1.0)  # 假设最大100亿
        win_rate_score = result['win_rate'] / 100.0
        
        composite_score = (
            total_return_score * 0.6 +
            risk_return_score * 0.3 +
            win_rate_score * 0.1
        )
        
        # 添加额外指标
        result['risk_return_ratio'] = risk_return_ratio
        result['composite_score'] = composite_score
        result['params'] = params.copy()
        
        return result
        
    except Exception as e:
        logger.error(f"参数组合回测失败: {params}, 错误: {e}")
        # 返回失败结果
        return {
            'total_return': -100.0,
            'final_equity': INITIAL_BALANCE,
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_loss_ratio': 0.0,
            'max_drawdown': 100.0,
            'avg_exposure': 0.0,
            'pyramid_trades': 0,
            'risk_return_ratio': -1.0,
            'composite_score': 0.0,
            'params': params,
            'error': str(e)
        }

def generate_parameter_combinations() -> List[Dict[str, Any]]:
    """
    生成所有参数组合
    
    返回:
        参数组合列表
    """
    param_names = list(PARAM_SEARCH_SPACE.keys())
    param_values = list(PARAM_SEARCH_SPACE.values())
    
    combinations = []
    for values in itertools.product(*param_values):
        params = {
            'pyramid_enabled': True,  # 始终启用金字塔
            **dict(zip(param_names, values))
        }
        combinations.append(params)
    
    logger.info(f"生成了 {len(combinations)} 种参数组合")
    return combinations

def run_optimization() -> pd.DataFrame:
    """
    运行参数优化主流程
    
    返回:
        包含所有回测结果的DataFrame
    """
    logger.info("=" * 80)
    logger.info("🚀 开始金字塔参数优化")
    logger.info("=" * 80)
    
    # 1. 加载数据（单次加载，避免重复计算）
    logger.info("\n步骤1: 加载数据和生成预测信号...")
    klines, predictions = load_data_and_predictions()
    
    # 2. 生成参数组合
    logger.info("\n步骤2: 生成参数组合...")
    param_combinations = generate_parameter_combinations()
    
    # 3. 添加默认参数组合作为基准
    param_combinations.insert(0, DEFAULT_PARAMS)
    logger.info(f"总参数组合数（含基准）: {len(param_combinations)}")
    
    # 4. 运行回测
    logger.info("\n步骤3: 运行回测...")
    results = []
    
    for i, params in enumerate(param_combinations):
        logger.info(f"\n[{i+1}/{len(param_combinations)}] 测试参数组合:")
        logger.info(f"  盈利阈值: {params['pyramid_profit_threshold']*100:.1f}%")
        logger.info(f"  最小RR: {params['pyramid_min_rr']:.1f}")
        logger.info(f"  最小概率: {params['pyramid_min_prob']:.2f}")
        logger.info(f"  最大加仓次数: {params['pyramid_max_count']}")
        logger.info(f"  最小K线间隔: {params['pyramid_min_bars']}")
        logger.info(f"  总敞口上限: {params['max_total_exposure']:.1f}倍")
        
        result = run_single_backtest(klines, predictions, params)
        results.append(result)
        
        # 显示关键结果
        logger.info(f"  结果: 收益率={result['total_return']:.2f}%, "
                   f"风险收益比={result.get('risk_return_ratio', 0):.2f}, "
                   f"综合评分={result.get('composite_score', 0):.4f}")
    
    # 5. 转换为DataFrame并排序
    logger.info("\n步骤4: 分析结果...")
    results_df = pd.DataFrame(results)
    
    # 按综合评分排序
    results_df = results_df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    
    return results_df

def analyze_results(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    分析优化结果
    
    参数:
        results_df: 回测结果DataFrame
        
    返回:
        分析结果字典
    """
    logger.info("=" * 80)
    logger.info("📊 优化结果分析")
    logger.info("=" * 80)
    
    if len(results_df) == 0:
        logger.error("没有可分析的结果")
        return {}
    
    # 1. 最佳参数组合
    best_result = results_df.iloc[0]
    best_params = best_result['params']
    
    logger.info("\n🎯 最佳参数组合 (综合评分最高):")
    logger.info(f"  综合评分: {best_result['composite_score']:.4f}")
    logger.info(f"  总收益率: {best_result['total_return']:.2f}%")
    logger.info(f"  风险收益比: {best_result.get('risk_return_ratio', 0):.2f}")
    logger.info(f"  胜率: {best_result['win_rate']:.2f}%")
    logger.info(f"  盈亏比: {best_result['profit_loss_ratio']:.2f}")
    logger.info(f"  最大回撤: {best_result['max_drawdown']:.2f}%")
    logger.info(f"  加仓交易数: {best_result['pyramid_trades']}")
    
    logger.info("\n📋 最佳参数配置:")
    for key, value in best_params.items():
        if key != 'pyramid_enabled':
            logger.info(f"  {key}: {value}")
    
    # 2. 基准参数结果（默认配置）
    baseline_mask = results_df.apply(lambda row: all(
        row['params'].get(k) == v for k, v in DEFAULT_PARAMS.items() if k != 'pyramid_enabled'
    ), axis=1)
    
    if baseline_mask.any():
        baseline_result = results_df[baseline_mask].iloc[0]
        logger.info("\n📈 基准参数结果 (当前配置):")
        logger.info(f"  综合评分: {baseline_result['composite_score']:.4f}")
        logger.info(f"  总收益率: {baseline_result['total_return']:.2f}%")
        logger.info(f"  风险收益比: {baseline_result.get('risk_return_ratio', 0):.2f}")
        
        # 计算提升幅度
        improvement_pct = (best_result['composite_score'] - baseline_result['composite_score']) / baseline_result['composite_score'] * 100
        logger.info(f"  综合评分提升: {improvement_pct:.1f}%")
    
    # 3. 参数敏感性分析
    logger.info("\n🔬 参数敏感性分析:")
    param_importance = {}
    
    for param_name in PARAM_SEARCH_SPACE.keys():
        param_values = []
        scores = []
        
        for _, row in results_df.iterrows():
            param_value = row['params'].get(param_name)
            if param_value is not None:
                param_values.append(param_value)
                scores.append(row['composite_score'])
        
        if param_values:
            # 计算每个参数值的平均得分
            unique_values = sorted(set(param_values))
            value_scores = []
            for val in unique_values:
                val_scores = [s for p, s in zip(param_values, scores) if p == val]
                if val_scores:
                    value_scores.append((val, np.mean(val_scores)))
            
            if value_scores:
                # 找到最佳参数值
                best_val, best_score = max(value_scores, key=lambda x: x[1])
                worst_val, worst_score = min(value_scores, key=lambda x: x[1])
                
                sensitivity = (best_score - worst_score) / worst_score * 100 if worst_score > 0 else 0
                param_importance[param_name] = sensitivity
                
                logger.info(f"  {param_name}:")
                logger.info(f"    最佳值: {best_val} (得分: {best_score:.4f})")
                logger.info(f"    最差值: {worst_val} (得分: {worst_score:.4f})")
                logger.info(f"    敏感度: {sensitivity:.1f}%")
    
    # 4. 保存结果
    output_dir = Path('backtest/parameter_optimization')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存详细结果
    detailed_results = results_df.copy()
    detailed_results['params_json'] = detailed_results['params'].apply(json.dumps)
    detailed_results = detailed_results.drop(columns=['params'])
    
    output_file = output_dir / f'pyramid_optimization_results_{timestamp}.csv'
    detailed_results.to_csv(output_file, index=False)
    logger.info(f"\n📁 详细结果已保存至: {output_file}")
    
    # 保存最佳参数配置
    best_config_file = output_dir / f'best_pyramid_config_{timestamp}.json'
    with open(best_config_file, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    logger.info(f"📁 最佳参数配置已保存至: {best_config_file}")
    
    # 5. 生成建议
    logger.info("\n💡 优化建议:")
    
    # 检查是否有明显改进
    if 'improvement_pct' in locals() and improvement_pct > 5.0:
        logger.info("  1. ✅ 发现显著更好的参数组合，建议更新配置")
        logger.info(f"  2. 📈 预期提升: {improvement_pct:.1f}%")
    else:
        logger.info("  1. ⚠️  当前参数配置已接近最优，提升空间有限")
    
    # 参数调整建议
    logger.info("  3. 🎛️  参数调整方向:")
    for param_name, sensitivity in sorted(param_importance.items(), key=lambda x: x[1], reverse=True):
        if sensitivity > 10.0:  # 高敏感度参数
            current_val = DEFAULT_PARAMS.get(param_name)
            best_val = best_params.get(param_name)
            if current_val != best_val:
                logger.info(f"    - {param_name}: {current_val} → {best_val} (敏感度: {sensitivity:.1f}%)")
    
    return {
        'best_params': best_params,
        'best_result': best_result.to_dict(),
        'param_importance': param_importance,
        'output_files': {
            'detailed_results': str(output_file),
            'best_config': str(best_config_file)
        }
    }

def main():
    """主函数"""
    logger.info(__doc__)
    
    try:
        # 运行优化
        results_df = run_optimization()
        
        # 分析结果
        analysis = analyze_results(results_df)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ 参数优化完成!")
        logger.info("=" * 80)
        
        # 显示最佳参数配置（可直接复制到config.yaml）
        if analysis:
            best_params = analysis.get('best_params', {})
            logger.info("\n📋 最佳参数配置 (可直接复制到config.yaml的enhanced部分):")
            logger.info("  # 金字塔加仓配置 (优化后)")
            for key, value in best_params.items():
                if key != 'pyramid_enabled':
                    logger.info(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"参数优化失败: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()