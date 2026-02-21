#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测引擎模块 - 包含所有回测核心逻辑

本模块提供完整的回测功能,包括:
- 动态敞口计算
- 多层风控回测
- 结果统计和输出
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np
from pathlib import Path
import logging
from datetime import datetime
from btc_quant.config import load_config
from btc_quant.data import load_klines
from btc_quant.features import build_features_and_labels
from btc_quant.risk_reward_model import TwoStageRiskRewardStrategy

# 日志配置
log_file = Path('../logs') / f'backtest_2024_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(levelname)s-%(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger=logging.getLogger(__name__)
logger.info(f"日志文件: {log_file.absolute()}")

# 配置常量
BACKTEST_START='2025-01-01T00:00:00Z'
BACKTEST_END='2026-02-20T23:59:59Z'
MODEL_DIR=Path('models/final_2024_dynamic')
INITIAL_BALANCE=1000.0
MAX_EXPOSURE=10.0
STOP_LOSS_PCT=-0.03
MAX_DAILY_LOSS_PCT=-0.20
MAX_DRAWDOWN_PAUSE=0.10
USE_TRAILING_STOP=True
RR_THRESHOLD=2.5
PROB_THRESHOLD=0.70
OUTPUT_DIR=Path('backtest_results')

def calculate_dynamic_exposure(predicted_rr,direction_prob,current_drawdown=0,consecutive_losses=0,max_exposure=10.0):
    """动态敞口计算"""
    rr_factor=min(predicted_rr/2.5,2.0)
    prob_factor=max((direction_prob-0.5)/0.5,0)
    base_exposure=2.0+rr_factor*3.0+prob_factor*3.0
    if current_drawdown>0.02:
        drawdown_penalty=max(0.3,1.0-(current_drawdown-0.02)*15)
    else:
        drawdown_penalty=1.0
    if consecutive_losses>=2:
        loss_penalty=max(0.2,1.0-min(consecutive_losses-1,5)*0.15)
    else:
        loss_penalty=1.0
    final_exposure=base_exposure*drawdown_penalty*loss_penalty
    return np.clip(final_exposure,1.0,max_exposure)

def advanced_risk_backtest(klines,predictions,initial_balance=1000.0,max_exposure=10.0,stop_loss_pct=-0.03,max_daily_loss_pct=-0.20,max_drawdown_pause=0.10,use_trailing_stop=True):
    """多层风控回测引擎"""
    equity,peak_equity,position,trades=initial_balance,initial_balance,None,[]
    daily_start_equity,current_date=initial_balance,None
    daily_loss_paused,drawdown_paused,consecutive_losses=False,False,0
    for i in range(len(predictions)):
        current_time,current_price=klines.iloc[i]['open_time'],klines.iloc[i]['close']
        current_day=pd.Timestamp(current_time).date()
        if current_date!=current_day:
            current_date,daily_start_equity=current_day,equity
            if daily_loss_paused:
                daily_loss_paused=False
                logger.info(f"[{current_time}]新的一天,恢复交易(每日亏损暂停已解除)")
            if drawdown_paused:
                drawdown_paused,current_drawdown=False,0
                logger.info(f"[{current_time}]新的一天,恢复交易(回撤暂停已解除)")
        current_drawdown=(peak_equity-equity)/peak_equity if peak_equity>0 else 0
        if position is not None:
            bars_held=i-position['entry_idx']
            if position['side']==1:
                price_change_pct=(current_price-position['entry_price'])/position['entry_price']
            else:
                price_change_pct=(position['entry_price']-current_price)/position['entry_price']
            current_pnl_pct=price_change_pct*position['exposure']
            if current_pnl_pct>position['peak_pnl_pct']:
                position['peak_pnl_pct'],position['peak_price']=current_pnl_pct,current_price
            should_close,close_reason,stop_loss_hit,trailing_stop_hit=False,"",False,False
            if current_pnl_pct<=stop_loss_pct:
                should_close,close_reason,stop_loss_hit=True,"固定止损",True
            elif use_trailing_stop and position['peak_pnl_pct']>0.01:
                if position['side']==1:
                    drawdown_from_peak=(position['peak_price']-current_price)/position['peak_price']
                else:
                    drawdown_from_peak=(current_price-position['peak_price'])/position['peak_price']
                if drawdown_from_peak>0.02:
                    should_close,close_reason,trailing_stop_hit=True,"追踪止损",True
            elif bars_held>=position['hold_period']:
                should_close,close_reason=True,"周期到期"
            if should_close:
                pnl=equity*current_pnl_pct
                equity+=pnl
                if equity>peak_equity:
                    peak_equity=equity
                if pnl<=0:
                    consecutive_losses+=1
                else:
                    consecutive_losses=0
                trades.append({'entry_time':klines.iloc[position['entry_idx']]['open_time'],'exit_time':current_time,'side':'long' if position['side']==1 else 'short','entry_price':position['entry_price'],'exit_price':current_price,'exposure':position['exposure'],'price_change_pct':price_change_pct*100,'pnl_pct':current_pnl_pct*100,'pnl':pnl,'equity_after':equity,'close_reason':close_reason,'stop_loss_hit':stop_loss_hit,'trailing_stop_hit':trailing_stop_hit,'consecutive_losses':consecutive_losses})
                position=None
                daily_loss_pct=(equity-daily_start_equity)/daily_start_equity
                if daily_loss_pct<max_daily_loss_pct:
                    daily_loss_paused=True
                    logger.warning(f"[{current_time}]触发每日最大亏损限制:{daily_loss_pct*100:.2f}%,暂停交易至明日")
                current_drawdown=(peak_equity-equity)/peak_equity if peak_equity>0 else 0
                if current_drawdown>max_drawdown_pause:
                    drawdown_paused=True
                    logger.warning(f"[{current_time}]触发回撤暂停:{current_drawdown*100:.2f}%,暂停交易至明日")
        if position is None and predictions.iloc[i]['should_trade']:
            if daily_loss_paused or drawdown_paused:
                continue
            exposure=calculate_dynamic_exposure(predictions.iloc[i]['predicted_rr'],predictions.iloc[i]['direction_prob'],current_drawdown,consecutive_losses,max_exposure)
            position={'side':predictions.iloc[i]['direction'],'entry_price':current_price,'entry_idx':i,'hold_period':int(predictions.iloc[i]['holding_period']),'exposure':exposure,'peak_pnl_pct':0,'peak_price':current_price}
    if position is not None:
        current_price=klines.iloc[-1]['close']
        if position['side']==1:
            price_change_pct=(current_price-position['entry_price'])/position['entry_price']
        else:
            price_change_pct=(position['entry_price']-current_price)/position['entry_price']
        current_pnl_pct=price_change_pct*position['exposure']
        pnl=equity*current_pnl_pct
        equity+=pnl
        trades.append({'entry_time':klines.iloc[position['entry_idx']]['open_time'],'exit_time':klines.iloc[-1]['open_time'],'side':'long' if position['side']==1 else 'short','entry_price':position['entry_price'],'exit_price':current_price,'exposure':position['exposure'],'price_change_pct':price_change_pct*100,'pnl_pct':current_pnl_pct*100,'pnl':pnl,'equity_after':equity,'close_reason':'强制平仓','stop_loss_hit':False,'trailing_stop_hit':False,'consecutive_losses':consecutive_losses})
    total_return=(equity/initial_balance-1)*100
    if len(trades)>0:
        trades_df=pd.DataFrame(trades)
        winning_trades,losing_trades=(trades_df['pnl']>0).sum(),(trades_df['pnl']<=0).sum()
        win_rate=winning_trades/len(trades)*100
        avg_win=trades_df[trades_df['pnl']>0]['pnl'].mean() if winning_trades>0 else 0
        avg_loss=abs(trades_df[trades_df['pnl']<=0]['pnl'].mean()) if losing_trades>0 else 0
        profit_loss_ratio=avg_win/avg_loss if avg_loss>0 else 0
        peak=trades_df['equity_after'].expanding().max()
        drawdown_series=(peak-trades_df['equity_after'])/peak*100
        max_drawdown=drawdown_series.max()
        stop_loss_count=len(trades_df[trades_df['stop_loss_hit']==True])
        trailing_stop_count=len(trades_df[trades_df['trailing_stop_hit']==True])
        avg_exposure=trades_df['exposure'].mean()
        max_consecutive_losses=trades_df['consecutive_losses'].max()
    else:
        win_rate,profit_loss_ratio,max_drawdown,stop_loss_count,trailing_stop_count,avg_exposure,max_consecutive_losses,trades_df=0,0,0,0,0,0,0,None
    return {'total_return':total_return,'final_equity':equity,'total_trades':len(trades),'win_rate':win_rate,'profit_loss_ratio':profit_loss_ratio,'max_drawdown':max_drawdown,'stop_loss_count':stop_loss_count,'trailing_stop_count':trailing_stop_count,'avg_exposure':avg_exposure,'max_consecutive_losses':max_consecutive_losses,'trades':trades_df}

def run_backtest():
    """主回测流程"""
    logger.info("="*80)
    logger.info("最终版回测-2024年模型-复利+动态敞口")
    logger.info("="*80)
    logger.info(f"\n回测时间范围:{BACKTEST_START}至{BACKTEST_END}")
    cfg=load_config()
    logger.info("\n加载K线数据...")
    klines_all=load_klines(cfg)
    klines_all['close_time']=pd.to_datetime(klines_all['close_time'])
    backtest_start_ts,backtest_end_ts=pd.Timestamp(BACKTEST_START),pd.Timestamp(BACKTEST_END)
    klines_backtest=klines_all[(klines_all['close_time']>=backtest_start_ts)&(klines_all['close_time']<=backtest_end_ts)].reset_index(drop=True)
    logger.info(f"回测集K线数量:{len(klines_backtest)}")
    logger.info(f"回测集时间范围:{klines_backtest['close_time'].min()}至{klines_backtest['close_time'].max()}")
    logger.info("\n构建特征...")
    feature_label_data=build_features_and_labels(cfg,klines_backtest)
    X_backtest_full=feature_label_data.features.reset_index(drop=True)
    min_len=min(len(X_backtest_full),len(klines_backtest))
    X_backtest_full=X_backtest_full.iloc[:min_len]
    klines_backtest=klines_backtest.iloc[:min_len].reset_index(drop=True)
    logger.info(f"对齐后样本数:{len(X_backtest_full)}")
    logger.info(f"\n加载模型:{MODEL_DIR}")
    strategy=TwoStageRiskRewardStrategy()
    strategy.load(MODEL_DIR)
    with open(MODEL_DIR/'top30_features.txt','r') as f:
        top_30_features=[line.strip() for line in f.readlines()]
    logger.info(f"特征数量:{len(top_30_features)}")
    X_backtest_top30=X_backtest_full[top_30_features]
    logger.info("\n生成预测信号...")
    predictions_dict=strategy.predict(X_backtest_top30,rr_threshold=RR_THRESHOLD,prob_threshold=PROB_THRESHOLD)
    predictions=pd.DataFrame({'predicted_rr':predictions_dict['predicted_rr'],'direction':predictions_dict['direction'],'holding_period':predictions_dict['holding_period'].clip(1,30),'direction_prob':predictions_dict['direction_prob'],'should_trade':predictions_dict['should_trade']})
    logger.info(f"总样本数:{len(predictions)}")
    logger.info(f"应交易样本:{predictions['should_trade'].sum()}")
    logger.info(f"交易比例:{predictions['should_trade'].sum()/len(predictions)*100:.2f}%")
    logger.info("\n"+"="*80)
    logger.info("运行回测(复利+动态敞口+多层风控)")
    logger.info("="*80)
    result=advanced_risk_backtest(klines=klines_backtest,predictions=predictions,initial_balance=INITIAL_BALANCE,max_exposure=MAX_EXPOSURE,stop_loss_pct=STOP_LOSS_PCT,max_daily_loss_pct=MAX_DAILY_LOSS_PCT,max_drawdown_pause=MAX_DRAWDOWN_PAUSE,use_trailing_stop=USE_TRAILING_STOP)
    logger.info("\n"+"="*80)
    logger.info("回测结果")
    logger.info("="*80)
    logger.info(f"\n核心指标:")
    logger.info(f"  总收益率:{result['total_return']:.2f}%")
    logger.info(f"  最终权益:{result['final_equity']:.2f}USDT")
    logger.info(f"  交易数:{result['total_trades']}笔")
    logger.info(f"  胜率:{result['win_rate']:.2f}%")
    logger.info(f"  盈亏比:{result['profit_loss_ratio']:.2f}")
    logger.info(f"\n风险指标:")
    logger.info(f"  最大回撤:{result['max_drawdown']:.2f}%")
    logger.info(f"  平均敞口:{result['avg_exposure']:.2f}倍")
    logger.info(f"  最大连续亏损:{result['max_consecutive_losses']}笔")
    logger.info(f"\n风控统计:")
    logger.info(f"  固定止损触发:{result['stop_loss_count']}次")
    logger.info(f"  追踪止损触发:{result['trailing_stop_count']}次")
    if result['trades'] is not None:
        OUTPUT_DIR.mkdir(exist_ok=True)
        timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file=OUTPUT_DIR/f'final_2024_dynamic_{timestamp}.csv'
        result['trades'].to_csv(output_file,index=False)
        logger.info(f"\n交易记录已保存:{output_file}")
    logger.info("\n"+"="*80)
    logger.info("回测完成!")
    logger.info("="*80)
