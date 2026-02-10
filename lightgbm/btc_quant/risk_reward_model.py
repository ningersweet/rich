#!/usr/bin/env python3
"""
两阶段盈亏比预测模型

阶段1：盈亏比预测模型（筛选高质量交易机会）
阶段2：方向+周期预测模型（在高质量机会中预测方向和持有时间）

目标：提升胜率至45%+
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Tuple, Optional
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class RiskRewardModel:
    """
    阶段1：盈亏比预测模型
    
    预测每笔交易的盈亏比，用于筛选高质量交易机会
    """
    
    def __init__(self, params: Optional[Dict] = None, loss_function: str = 'rmse'):
        """
        初始化盈亏比预测模型
        
        Args:
            params: LightGBM参数
            loss_function: 损失函数类型 ('rmse', 'fair', 'huber', 'quantile')
        """
        # DeepSeek第二周优化：支持多种损失函数
        if loss_function == 'fair':
            metric = 'fair'
        elif loss_function == 'huber':
            metric = 'huber'
        elif loss_function == 'quantile':
            metric = 'quantile'
        else:
            metric = 'rmse'
        
        self.params = params or {
            'objective': 'regression',
            'metric': metric,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
        }
        self.loss_function = loss_function
        self.model = None
        self.feature_importance = None
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict:
        """
        训练盈亏比预测模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签（盈亏比）
            X_val: 验证特征
            y_val: 验证标签
        
        Returns:
            训练指标字典
        """
        logger.info("开始训练盈亏比预测模型...")
        logger.info(f"训练样本: {len(X_train)}, 特征数: {X_train.shape[1]}")
        
        # 创建LightGBM数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val) if X_val is not None else None
        
        # 训练模型
        callbacks = [lgb.log_evaluation(period=50)]
        if valid_data is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=20))
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=500,
            valid_sets=[valid_data] if valid_data else None,
            callbacks=callbacks,
        )
        
        # 特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        logger.info("\n特征重要性 Top 10:")
        for idx, row in self.feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.0f}")
        
        # 评估
        metrics = {}
        y_train_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
        metrics['train_rmse'] = train_rmse
        
        if X_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_rmse = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
            metrics['val_rmse'] = val_rmse
            logger.info(f"训练RMSE: {train_rmse:.3f}, 验证RMSE: {val_rmse:.3f}")
        else:
            logger.info(f"训练RMSE: {train_rmse:.3f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测盈亏比"""
        if self.model is None:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    def save(self, path: Path):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未训练")
        self.model.save_model(str(path))
        logger.info(f"盈亏比模型已保存: {path}")
    
    def load(self, path: Path):
        """加载模型"""
        self.model = lgb.Booster(model_file=str(path))
        logger.info(f"盈亏比模型已加载: {path}")


class DirectionPeriodModel:
    """
    阶段2：方向+周期预测模型
    
    在高盈亏比的交易机会中，预测方向和持有周期
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        初始化方向+周期预测模型
        
        Args:
            params: LightGBM参数
        """
        self.direction_params = params or {
            'objective': 'multiclass',
            'num_class': 3,  # 做多(2)、观望(1)、做空(0)
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
        }
        
        self.period_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
        }
        
        self.direction_model = None
        self.period_model = None
        self.direction_importance = None
        self.period_importance = None
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_direction_train: pd.Series,
        y_period_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_direction_val: Optional[pd.Series] = None,
        y_period_val: Optional[pd.Series] = None,
    ) -> Dict:
        """
        训练方向和周期预测模型
        
        Args:
            X_train: 训练特征（包含预测的盈亏比）
            y_direction_train: 方向标签（-1, 0, 1）
            y_period_train: 持有周期标签
            X_val: 验证特征
            y_direction_val: 验证方向标签
            y_period_val: 验证周期标签
        
        Returns:
            训练指标字典
        """
        logger.info("\n开始训练方向预测模型...")
        
        # 将方向标签映射到0,1,2
        y_direction_train_mapped = y_direction_train + 1  # -1,0,1 -> 0,1,2
        if y_direction_val is not None:
            y_direction_val_mapped = y_direction_val + 1
        
        # 训练方向模型
        train_data_dir = lgb.Dataset(X_train, label=y_direction_train_mapped)
        valid_data_dir = lgb.Dataset(X_val, label=y_direction_val_mapped) if X_val is not None else None
        
        callbacks = [lgb.log_evaluation(period=50)]
        if valid_data_dir is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=20))
        
        self.direction_model = lgb.train(
            self.direction_params,
            train_data_dir,
            num_boost_round=500,
            valid_sets=[valid_data_dir] if valid_data_dir else None,
            callbacks=callbacks,
        )
        
        self.direction_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.direction_model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        logger.info("\n方向模型特征重要性 Top 10:")
        for idx, row in self.direction_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.0f}")
        
        # 训练周期模型（只用有交易信号的样本）
        logger.info("\n开始训练持有周期预测模型...")
        mask_train = y_direction_train != 0
        X_train_active = X_train[mask_train]
        y_period_train_active = y_period_train[mask_train]
        
        if X_val is not None:
            mask_val = y_direction_val != 0
            X_val_active = X_val[mask_val]
            y_period_val_active = y_period_val[mask_val]
        else:
            X_val_active = None
            y_period_val_active = None
        
        logger.info(f"训练样本（有交易信号）: {len(X_train_active)}")
        
        train_data_period = lgb.Dataset(X_train_active, label=y_period_train_active)
        valid_data_period = lgb.Dataset(X_val_active, label=y_period_val_active) if X_val_active is not None else None
        
        callbacks = [lgb.log_evaluation(period=50)]
        if valid_data_period is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=20))
        
        self.period_model = lgb.train(
            self.period_params,
            train_data_period,
            num_boost_round=500,
            valid_sets=[valid_data_period] if valid_data_period else None,
            callbacks=callbacks,
        )
        
        self.period_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.period_model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        logger.info("\n周期模型特征重要性 Top 10:")
        for idx, row in self.period_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.0f}")
        
        # 评估
        metrics = {}
        
        # 方向准确率
        y_direction_train_pred = self.direction_model.predict(X_train).argmax(axis=1) - 1  # 0,1,2 -> -1,0,1
        train_acc = (y_direction_train_pred == y_direction_train).mean()
        metrics['train_direction_acc'] = train_acc
        
        if X_val is not None:
            y_direction_val_pred = self.direction_model.predict(X_val).argmax(axis=1) - 1
            val_acc = (y_direction_val_pred == y_direction_val).mean()
            metrics['val_direction_acc'] = val_acc
            logger.info(f"方向准确率 - 训练: {train_acc:.3f}, 验证: {val_acc:.3f}")
        else:
            logger.info(f"方向准确率 - 训练: {train_acc:.3f}")
        
        # 周期RMSE
        y_period_train_pred = self.period_model.predict(X_train_active)
        train_period_rmse = np.sqrt(np.mean((y_period_train_active - y_period_train_pred) ** 2))
        metrics['train_period_rmse'] = train_period_rmse
        
        if X_val_active is not None:
            y_period_val_pred = self.period_model.predict(X_val_active)
            val_period_rmse = np.sqrt(np.mean((y_period_val_active - y_period_val_pred) ** 2))
            metrics['val_period_rmse'] = val_period_rmse
            logger.info(f"周期RMSE - 训练: {train_period_rmse:.1f}, 验证: {val_period_rmse:.1f}")
        else:
            logger.info(f"周期RMSE - 训练: {train_period_rmse:.1f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        预测方向和周期
        
        Returns:
            direction_pred: 预测方向 (-1, 0, 1)
            direction_proba: 方向概率 [做空概率, 观望概率, 做多概率]
            period_pred: 预测持有周期
        """
        if self.direction_model is None or self.period_model is None:
            raise ValueError("模型未训练")
        
        # 方向预测
        direction_proba = self.direction_model.predict(X)  # shape: (n, 3)
        direction_pred = direction_proba.argmax(axis=1) - 1  # 0,1,2 -> -1,0,1
        
        # 周期预测
        period_pred = self.period_model.predict(X)
        period_pred = np.clip(period_pred, 1, 50)  # 限制在1-50根K线
        
        return direction_pred, direction_proba, period_pred
    
    def save(self, direction_path: Path, period_path: Path):
        """保存模型"""
        if self.direction_model is None or self.period_model is None:
            raise ValueError("模型未训练")
        self.direction_model.save_model(str(direction_path))
        self.period_model.save_model(str(period_path))
        logger.info(f"方向模型已保存: {direction_path}")
        logger.info(f"周期模型已保存: {period_path}")
    
    def load(self, direction_path: Path, period_path: Path):
        """加载模型"""
        self.direction_model = lgb.Booster(model_file=str(direction_path))
        self.period_model = lgb.Booster(model_file=str(period_path))
        logger.info(f"方向模型已加载: {direction_path}")
        logger.info(f"周期模型已加载: {period_path}")


class TwoStageRiskRewardStrategy:
    """
    两阶段盈亏比驱动策略
    
    完整的训练和预测流程
    """
    
    def __init__(self, loss_function: str = 'rmse'):
        """
        初始化两阶段策略
        
        Args:
            loss_function: 损失函数类型 ('rmse', 'fair', 'huber', 'quantile')
        """
        self.loss_function = loss_function
        self.rr_model = RiskRewardModel(loss_function=loss_function)
        self.dp_model = DirectionPeriodModel()
    
    def train(
        self,
        X_train: pd.DataFrame,
        labels_train: pd.DataFrame,
        X_val: pd.DataFrame,
        labels_val: pd.DataFrame,
        rr_threshold: float = 2.0,
    ) -> Dict:
        """
        训练两阶段模型
        
        Args:
            X_train: 训练特征
            labels_train: 训练标签（包含direction, risk_reward, holding_period）
            X_val: 验证特征
            labels_val: 验证标签
            rr_threshold: 盈亏比阈值，只训练高于此值的样本
        
        Returns:
            训练指标字典
        """
        logger.info("=" * 80)
        logger.info("开始两阶段模型训练")
        logger.info("=" * 80)
        
        # 阶段1：训练盈亏比预测模型（使用所有有交易信号的样本）
        logger.info("\n【阶段1】训练盈亏比预测模型")
        
        # 筛选有交易信号的样本
        mask_train_active = labels_train['direction'] != 0
        mask_val_active = labels_val['direction'] != 0
        
        X_train_active = X_train[mask_train_active]
        y_rr_train = labels_train.loc[mask_train_active, 'risk_reward']
        
        X_val_active = X_val[mask_val_active]
        y_rr_val = labels_val.loc[mask_val_active, 'risk_reward']
        
        logger.info(f"训练样本（有信号）: {len(X_train_active):,}")
        logger.info(f"验证样本（有信号）: {len(X_val_active):,}")
        
        stage1_metrics = self.rr_model.train(
            X_train_active, y_rr_train,
            X_val_active, y_rr_val
        )
        
        # 阶段2：训练方向+周期模型（只用高盈亏比的样本）
        logger.info(f"\n【阶段2】训练方向+周期模型（RR>{rr_threshold}）")
        
        # 预测训练集的盈亏比
        rr_pred_train = self.rr_model.predict(X_train)
        rr_pred_val = self.rr_model.predict(X_val)
        
        # 筛选高盈亏比样本
        mask_train_high_rr = rr_pred_train > rr_threshold
        mask_val_high_rr = rr_pred_val > rr_threshold
        
        X_train_high_rr = X_train[mask_train_high_rr].copy()
        X_val_high_rr = X_val[mask_val_high_rr].copy()
        
        # 添加预测的盈亏比作为特征
        X_train_high_rr['predicted_rr'] = rr_pred_train[mask_train_high_rr]
        X_val_high_rr['predicted_rr'] = rr_pred_val[mask_val_high_rr]
        
        y_direction_train = labels_train.loc[mask_train_high_rr, 'direction']
        y_period_train = labels_train.loc[mask_train_high_rr, 'holding_period']
        
        y_direction_val = labels_val.loc[mask_val_high_rr, 'direction']
        y_period_val = labels_val.loc[mask_val_high_rr, 'holding_period']
        
        logger.info(f"训练样本（高RR）: {len(X_train_high_rr):,}")
        logger.info(f"验证样本（高RR）: {len(X_val_high_rr):,}")
        
        stage2_metrics = self.dp_model.train(
            X_train_high_rr, y_direction_train, y_period_train,
            X_val_high_rr, y_direction_val, y_period_val
        )
        
        # 合并指标
        metrics = {**stage1_metrics, **stage2_metrics}
        
        logger.info("\n" + "=" * 80)
        logger.info("两阶段模型训练完成！")
        logger.info("=" * 80)
        
        return metrics
    
    def predict(
        self,
        X: pd.DataFrame,
        rr_threshold: float = 2.0,
        prob_threshold: float = 0.65,
    ) -> pd.DataFrame:
        """
        两阶段预测
        
        Args:
            X: 特征
            rr_threshold: 盈亏比阈值
            prob_threshold: 方向概率阈值
        
        Returns:
            预测结果DataFrame
        """
        # 阶段1：预测盈亏比
        rr_pred = self.rr_model.predict(X)
        
        # 初始化结果
        results = pd.DataFrame({
            'predicted_rr': rr_pred,
            'direction': 0,
            'direction_prob': 0.0,
            'holding_period': 0,
            'should_trade': False,
        }, index=X.index)
        
        # 筛选高盈亏比样本
        mask_high_rr = rr_pred > rr_threshold
        
        if mask_high_rr.sum() > 0:
            # 阶段2：预测方向和周期
            X_high_rr = X[mask_high_rr].copy()
            X_high_rr['predicted_rr'] = rr_pred[mask_high_rr]
            
            direction_pred, direction_proba, period_pred = self.dp_model.predict(X_high_rr)
            
            # 提取最大概率
            max_prob = direction_proba.max(axis=1)
            
            # 更新结果
            results.loc[mask_high_rr, 'direction'] = direction_pred
            results.loc[mask_high_rr, 'direction_prob'] = max_prob
            results.loc[mask_high_rr, 'holding_period'] = period_pred
            
            # 决定是否交易：方向非观望 且 概率足够高
            should_trade = (direction_pred != 0) & (max_prob > prob_threshold)
            results.loc[mask_high_rr, 'should_trade'] = should_trade
        
        return results
    
    def save(self, model_dir: Path):
        """保存所有模型"""
        model_dir.mkdir(parents=True, exist_ok=True)
        
        rr_path = model_dir / 'risk_reward_model.txt'
        direction_path = model_dir / 'direction_model.txt'
        period_path = model_dir / 'period_model.txt'
        
        self.rr_model.save(rr_path)
        self.dp_model.save(direction_path, period_path)
        
        logger.info(f"\n所有模型已保存到: {model_dir}")
    
    def load(self, model_dir: Path):
        """加载所有模型"""
        rr_path = model_dir / 'risk_reward_model.txt'
        direction_path = model_dir / 'direction_model.txt'
        period_path = model_dir / 'period_model.txt'
        
        self.rr_model.load(rr_path)
        self.dp_model.load(direction_path, period_path)
        
        logger.info(f"\n所有模型已从 {model_dir} 加载")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("两阶段盈亏比模型模块")
    print("请运行训练脚本进行完整训练")
