"""集成学习策略模块。

核心思路：
1. 训练多个不同类型的模型
2. 每个模型独立预测
3. 通过加权投票得到最终信号
4. 信号置信度 = 模型一致性
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from .config import Config


@dataclass
class EnsembleModel:
    """集成模型。"""
    
    # 主模型：LightGBM
    lgb_model: lgb.Booster
    
    # 辅助模型1：随机森林（擅长捕捉非线性关系）
    rf_model: Optional[RandomForestClassifier] = None
    
    # 辅助模型2：逻辑回归（擅长捕捉线性趋势）
    lr_model: Optional[LogisticRegression] = None
    
    # 特征名称
    feature_names: List[str] = None
    
    # 模型权重（归一化后和为1）
    weights: Dict[str, float] = None


@dataclass
class EnsemblePrediction:
    """集成预测结果。"""
    
    # 做多概率（可以是标量或数组）
    prob_long: np.ndarray
    
    # 做空概率
    prob_short: np.ndarray
    
    # 模型一致性（0-1，越高说明模型越一致）
    consistency: np.ndarray
    
    # 观望概率（可选）
    prob_neutral: Optional[np.ndarray] = None
    
    # 各模型的原始预测（可选）
    individual_predictions: Optional[Dict[str, np.ndarray]] = None


def train_ensemble_models(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Config,
) -> EnsembleModel:
    """训练集成模型。
    
    Args:
        X: 特征矩阵
        y: 标签（-1, 0, 1）
        cfg: 配置对象
    
    Returns:
        集成模型对象
    """
    # 清理数据：删除NaN和无穷大
    X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    model_cfg = cfg.model
    train_cfg = model_cfg.get("train", {})
    
    # 1. 训练主模型：LightGBM
    print("训练主模型：LightGBM...")
    lgb_params = dict(model_cfg.get("params", {}))
    lgb_params.update({
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'verbosity': -1,
    })
    
    # 类别映射：-1, 0, 1 -> 0, 1, 2
    y_mapped = y.map({-1: 0, 0: 1, 1: 2})
    
    # 时间序列交叉验证
    cv_folds = int(train_cfg.get("cv_folds", 3))
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    lgb_models = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_clean)):
        X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
        y_train, y_val = y_mapped.iloc[train_idx], y_mapped.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=300,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )
        lgb_models.append(model)
    
    # 使用最后一折的模型
    lgb_model = lgb_models[-1]
    
    # 2. 训练辅助模型1：随机森林（采样子集以加快速度）
    rf_model = None
    if train_cfg.get("use_random_forest", True):
        print("训练辅助模型：随机森林...")
        sample_size = min(50000, len(X_clean))
        sample_idx = np.random.choice(len(X_clean), sample_size, replace=False)
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=100,
            n_jobs=-1,
            random_state=42,
        )
        rf_model.fit(X_clean.iloc[sample_idx], y_mapped.iloc[sample_idx])
    
    # 3. 训练辅助模型2：逻辑回归（快速线性模型）
    lr_model = None
    if train_cfg.get("use_logistic_regression", True):
        print("训练辅助模型：逻辑回归...")
        lr_model = LogisticRegression(
            max_iter=500,
            multi_class='multinomial',
            n_jobs=-1,
            random_state=42,
        )
        lr_model.fit(X_clean, y_mapped)
    
    # 4. 确定模型权重
    weights = {
        'lgb': 0.6,  # LightGBM主导
        'rf': 0.25 if rf_model else 0,
        'lr': 0.15 if lr_model else 0,
    }
    
    # 归一化权重
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    return EnsembleModel(
        lgb_model=lgb_model,
        rf_model=rf_model,
        lr_model=lr_model,
        feature_names=list(X_clean.columns),
        weights=weights,
    )


def predict_ensemble(
    ensemble: EnsembleModel,
    X: pd.DataFrame,
) -> EnsemblePrediction:
    """使用集成模型进行预测（支持批量）。
    
    Args:
        ensemble: 集成模型
        X: 特征（单行或多行）
    
    Returns:
        集成预测结果
    """
    batch_size = len(X)
    
    # 1. LightGBM预测
    lgb_probs = ensemble.lgb_model.predict(X)  # shape: (n, 3)
    
    # 2. 随机森林预测
    if ensemble.rf_model:
        rf_probs = ensemble.rf_model.predict_proba(X)  # shape: (n, 3)
    else:
        rf_probs = lgb_probs
    
    # 3. 逻辑回归预测
    if ensemble.lr_model:
        lr_probs = ensemble.lr_model.predict_proba(X)  # shape: (n, 3)
    else:
        lr_probs = lgb_probs
    
    # 4. 加权合并
    weights = ensemble.weights
    ensemble_probs = (
        lgb_probs * weights['lgb'] +
        rf_probs * weights.get('rf', 0.0) +
        lr_probs * weights.get('lr', 0.0)
    )
    
    # 5. 解析概率（假设类别顺序为 [-1, 0, 1]）
    prob_short = ensemble_probs[:, 0]  # -1
    prob_flat = ensemble_probs[:, 1]   # 0
    prob_long = ensemble_probs[:, 2]   # 1
    
    # 6. 计算模型一致性
    # 每个模型的预测类别
    lgb_class = np.argmax(lgb_probs, axis=1)
    rf_class = np.argmax(rf_probs, axis=1) if ensemble.rf_model else lgb_class
    lr_class = np.argmax(lr_probs, axis=1) if ensemble.lr_model else lgb_class
    
    # 一致性 = 预测类别相同的模型数 / 总模型数
    n_models = 1 + (1 if ensemble.rf_model else 0) + (1 if ensemble.lr_model else 0)
    consistency = np.zeros(batch_size)
    for i in range(batch_size):
        classes = [lgb_class[i]]
        if ensemble.rf_model:
            classes.append(rf_class[i])
        if ensemble.lr_model:
            classes.append(lr_class[i])
        # 计算最多类别的数量
        most_common_count = max([classes.count(c) for c in set(classes)])
        consistency[i] = most_common_count / n_models
    
    return EnsemblePrediction(
        prob_long=prob_long,
        prob_short=prob_short,
        consistency=consistency,
    )


def calculate_model_consistency(predictions: Dict[str, np.ndarray]) -> float:
    """计算模型预测的一致性。
    
    一致性定义：
    - 所有模型预测的类别相同 → 1.0
    - 模型预测完全分散 → 0.0
    
    Args:
        predictions: 各模型的概率预测（shape: (3,)）
    
    Returns:
        一致性评分（0-1）
    """
    if len(predictions) < 2:
        return 1.0
    
    # 找出每个模型的最高概率类别
    predicted_classes = [np.argmax(prob) for prob in predictions.values()]
    
    # 计算熵（分散程度）
    from collections import Counter
    counts = Counter(predicted_classes)
    n_models = len(predicted_classes)
    
    # 最大一致类别的占比
    max_consistency = max(counts.values()) / n_models
    
    return max_consistency


def save_ensemble_model(ensemble: EnsembleModel, cfg: Config, name: str = "ensemble_latest.pkl") -> Path:
    """保存集成模型。"""
    model_dir = Path(cfg.paths["model_dir"]).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / name
    
    joblib.dump({
        'lgb_model': ensemble.lgb_model,
        'rf_model': ensemble.rf_model,
        'lr_model': ensemble.lr_model,
        'feature_names': ensemble.feature_names,
        'weights': ensemble.weights,
    }, out_path)
    
    return out_path


def load_ensemble_model(cfg: Config, name: str = "ensemble_latest.pkl") -> EnsembleModel:
    """加载集成模型。"""
    model_dir = Path(cfg.paths["model_dir"]).expanduser().resolve()
    model_path = model_dir / name
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    data = joblib.load(model_path)
    
    return EnsembleModel(
        lgb_model=data['lgb_model'],
        rf_model=data.get('rf_model'),
        lr_model=data.get('lr_model'),
        feature_names=data['feature_names'],
        weights=data['weights'],
    )
