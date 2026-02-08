from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit

import lightgbm as lgb

from .config import Config
from .features import FeatureLabelData


@dataclass
class TrainedModel:
    model: lgb.LGBMClassifier
    feature_names: list[str]


def train_model(cfg: Config, data: FeatureLabelData) -> TrainedModel:
    """在 CPU 上训练 LightGBM 多分类模型。

    - 支持可选的贝叶斯超参数优化
    - 支持类别权重（缓解多/空样本稀缺问题）
    - 训练结束后基于特征重要性仅保留Top N个特征
    """

    X = data.features
    y = data.labels

    model_cfg = cfg.model
    params: dict = dict(model_cfg.get("params", {}))
    train_cfg = model_cfg.get("train", {})

    # 交叉验证折数
    cv_folds = int(train_cfg.get("cv_folds", 3))

    # 类别权重配置（标签取值为 -1/0/1）
    class_weight_cfg = train_cfg.get("class_weight")
    if class_weight_cfg:
        class_weight: dict[int, float] = {}
        for k, v in class_weight_cfg.items():
            try:
                class_weight[int(k)] = float(v)
            except Exception:  # noqa: BLE001
                continue
        if class_weight:
            params["class_weight"] = class_weight

    # 可选：先进行一次贝叶斯超参数优化
    if bool(train_cfg.get("use_bayesian_optimization", False)):
        bayes_n_iter = int(train_cfg.get("bayes_n_iter", 30))
        try:
            best_params = _bayesian_optimize_lgbm_params(X, y, params, cv_folds=cv_folds, n_iter=bayes_n_iter)
            params.update(best_params)
        except Exception as e:  # noqa: BLE001
            # 若优化失败，回退到原始参数
            print(f"[WARN] 贝叶斯超参数优化失败，使用原始参数。原因: {e}")

    clf = lgb.LGBMClassifier(**params)

    # 简单时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=cv_folds)

    all_reports: list[str] = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        report = classification_report(y_val, y_pred, digits=3)
        all_reports.append(f"Fold {fold} classification report:\n{report}")

    print("\n".join(all_reports))

    # 基于特征重要性选择Top N特征（使用最后一次训练得到的特征重要性）
    feature_names = list(X.columns)
    top_n = int(train_cfg.get("top_n_features", 0))
    if top_n > 0 and hasattr(clf, "feature_importances_"):
        importances = np.asarray(clf.feature_importances_, dtype="float64")
        if importances.sum() > 0:
            indices = np.argsort(importances)[::-1]
            k = min(top_n, len(feature_names))
            feature_names = [feature_names[i] for i in indices[:k]]

    # 使用全部数据重新训练最终模型（仅使用选出的特征）
    clf_final = lgb.LGBMClassifier(**params)
    clf_final.fit(X[feature_names], y)

    return TrainedModel(model=clf_final, feature_names=feature_names)


def _bayesian_optimize_lgbm_params(
    X: pd.DataFrame,
    y: pd.Series,
    base_params: dict,
    cv_folds: int,
    n_iter: int = 30,
) -> dict:
    """使用贝叶斯优化搜索 LightGBM 部分超参数。

    注意：仅在 config.model.train.use_bayesian_optimization=true 时被调用。
    """

    try:
        from skopt import BayesSearchCV
        from skopt.space import Real, Integer
    except ImportError as e:  # pragma: no cover - 依赖缺失时直接报错
        raise RuntimeError("需要安装 scikit-optimize 才能使用贝叶斯超参数优化") from e

    # 搜索空间（围绕现有参数做局部搜索）
    search_spaces = {
        "num_leaves": Integer(20, 100),
        "max_depth": Integer(4, 10),
        "learning_rate": Real(0.01, 0.1, prior="log-uniform"),
        "n_estimators": Integer(100, 500),
        "reg_alpha": Real(0.01, 0.3, prior="log-uniform"),
        "reg_lambda": Real(0.01, 0.3, prior="log-uniform"),
    }

    estimator = lgb.LGBMClassifier(**base_params)
    cv = TimeSeriesSplit(n_splits=cv_folds)

    opt = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_spaces,
        n_iter=n_iter,
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )

    opt.fit(X, y)
    return opt.best_params_

def save_model(cfg: Config, trained: TrainedModel, name: str = "model_latest.pkl") -> Path:
    model_dir = Path(cfg.paths["model_dir"]).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / name
    joblib.dump({"model": trained.model, "feature_names": trained.feature_names}, out_path)
    return out_path


def load_model(cfg: Config, name: str = "model_latest.pkl") -> TrainedModel:
    model_dir = Path(cfg.paths["model_dir"]).expanduser().resolve()
    path = model_dir / name
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在: {path}")
    obj = joblib.load(path)
    return TrainedModel(model=obj["model"], feature_names=obj["feature_names"])


def predict_proba(trained: TrainedModel, features: pd.DataFrame) -> np.ndarray:
    """对给定特征矩阵输出各类别概率。"""

    X = features[trained.feature_names]
    return trained.model.predict_proba(X)
