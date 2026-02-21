#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测引擎模块 - 金字塔加仓版本

本模块在原有回测基础上增加了金字塔加仓功能:
- 支持等比例加仓（最多3次）
- 盈利>1%后触发加仓
- 方向一致且信号质量高
- 统一平仓管理
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
log_file = Path('../logs') / f'backtest_pyramid_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
MAX_TOTAL_EXPOSURE=15.0  # 总敞口上限（含加仓）
STOP_LOSS_PCT=-0.03
MAX_DAILY_LOSS_PCT=-0.20
MAX_DRAWDOWN_PAUSE=0.10
USE_TRAILING_STOP=True
RR_THRESHOLD=2.5
PROB_THRESHOLD=0.70
OUTPUT_DIR=Path('backtest_results')

# 加仓配置
PYRAMID_ENABLED=True
PYRAMID_PROFIT_THRESHOLD=0.01  # 盈利>1%后允许加仓
PYRAMID_MIN_RR=3.0  # 加仓信号盈亏比阈值
PYRAMID_MIN_PROB=0.75  # 加仓信号概率阈值
PYRAMID_MAX_COUNT=3  # 最多加仓次数
PYRAMID_MIN_BARS=5  # 距上次加仓最小K线数

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

def pyramid_backtest_with_compounding(klines,predictions,initial_balance=1000.0,max_total_exposure=15.0,stop_loss_pct=-0.03,max_daily_loss_pct=-0.20,max_drawdown_pause=0.10,use_trailing_stop=True):
    """金字塔加仓回测引擎"""
    equity,peak_equity=initial_balance,initial_balance
    positions=[]  # 多仓位列表
    trades=[]
    daily_start_equity,current_date=initial_balance,None
    daily_loss_paused,drawdown_paused,consecutive_losses=False,False,0
    last_pyramid_idx=-999
    
    for i in range(len(predictions)):
        current_time,current_price=klines.iloc[i]['open_time'],klines.iloc[i]['close']
        current_day=pd.Timestamp(current_time).date()
        
        # 每日重置
        if current_date!=current_day:
            current_date,daily_start_equity=current_day,equity
            if daily_loss_paused:
                daily_loss_paused=False
                logger.info(f"[{current_time}]新的一天,恢复交易(每日亏损暂停已解除)")
            if drawdown_paused:
                drawdown_paused,current_drawdown=False,0
                logger.info(f"[{current_time}]新的一天,恢复交易(回撤暂停已解除)")
        
        current_drawdown=(peak_equity-equity)/peak_equity if peak_equity>0 else 0
        
        # 持仓管理
        if len(positions)>0:
            # 计算总盈亏
            total_current_pnl_pct=0
            peak_pnl_pct=0
            
            for pos in positions:
                bars_held=i-pos['entry_idx']
                if pos['side']==1:
                    price_change_pct=(current_price-pos['entry_price'])/pos['entry_price']
                else:
                    price_change_pct=(pos['entry_price']-current_price)/pos['entry_price']
                
                current_pnl_pct=price_change_pct*pos['exposure']
                total_current_pnl_pct+=current_pnl_pct
                
                # 更新各仓位峰值
                if current_pnl_pct>pos['peak_pnl_pct']:
                    pos['peak_pnl_pct'],pos['peak_price']=current_pnl_pct,current_price
                
                peak_pnl_pct=max(peak_pnl_pct,pos['peak_pnl_pct'])
            
            # 统一平仓检查
            should_close,close_reason,stop_loss_hit,trailing_stop_hit=False,"",False,False
            
            # 1. 固定止损
            if total_current_pnl_pct<=stop_loss_pct:
                should_close,close_reason,stop_loss_hit=True,"固定止损",True
            
            # 2. 追踪止损（任一仓位盈利>1%后启用）
            elif use_trailing_stop and peak_pnl_pct>0.01:
                if positions[0]['side']==1:
                    drawdown_from_peak=(positions[0]['peak_price']-current_price)/positions[0]['peak_price']
                else:
                    drawdown_from_peak=(current_price-positions[0]['peak_price'])/positions[0]['peak_price']
                
                if drawdown_from_peak>0.02:
                    should_close,close_reason,trailing_stop_hit=True,"追踪止损",True
            
            # 3. 周期到期（以首仓为准）
            elif i-positions[0]['entry_idx']>=positions[0]['hold_period']:
                should_close,close_reason=True,"周期到期"
            
            # 平仓处理
            if should_close:
                total_pnl=equity*total_current_pnl_pct
                equity+=total_pnl
                
                if equity>peak_equity:
                    peak_equity=equity
                
                if total_pnl<=0:
                    consecutive_losses+=1
                else:
                    consecutive_losses=0
                
                # 记录交易（合并所有仓位）
                total_exposure=sum(p['exposure'] for p in positions)
                avg_entry=sum(p['entry_price']*p['exposure'] for p in positions)/total_exposure if total_exposure>0 else positions[0]['entry_price']
                
                trades.append({
                    'entry_time':klines.iloc[positions[0]['entry_idx']]['open_time'],
                    'exit_time':current_time,
                    'side':'long' if positions[0]['side']==1 else 'short',
                    'entry_price':avg_entry,
                    'exit_price':current_price,
                    'exposure':total_exposure,
                    'pyramid_count':len(positions),
                    'price_change_pct':((current_price-avg_entry)/avg_entry if positions[0]['side']==1 else (avg_entry-current_price)/avg_entry)*100,
                    'pnl_pct':total_current_pnl_pct*100,
                    'pnl':total_pnl,
                    'equity_after':equity,
                    'close_reason':close_reason,
                    'stop_loss_hit':stop_loss_hit,
                    'trailing_stop_hit':trailing_stop_hit,
                    'consecutive_losses':consecutive_losses
                })
                
                positions=[]
                last_pyramid_idx=-999
                
                # 熔断检查
                daily_loss_pct=(equity-daily_start_equity)/daily_start_equity
                if daily_loss_pct<max_daily_loss_pct:
                    daily_loss_paused=True
                    logger.warning(f"[{current_time}]触发每日最大亏损限制:{daily_loss_pct*100:.2f}%,暂停交易至明日")
                
                current_drawdown=(peak_equity-equity)/peak_equity if peak_equity>0 else 0
                if current_drawdown>max_drawdown_pause:
                    drawdown_paused=True
                    logger.warning(f"[{current_time}]触发回撤暂停:{current_drawdown*100:.2f}%,暂停交易至明日")
        
        # 开仓/加仓逻辑
        if predictions.iloc[i]['should_trade']:
            if daily_loss_paused or drawdown_paused:
                continue
            
            # 计算当前信号敞口
            new_exposure=calculate_dynamic_exposure(
                predictions.iloc[i]['predicted_rr'],
                predictions.iloc[i]['direction_prob'],
                current_drawdown,
                consecutive_losses,
                max_exposure=10.0
            )
            
            # 情况1：无持仓，开新仓
            if len(positions)==0:
                positions.append({
                    'side':predictions.iloc[i]['direction'],
                    'entry_price':current_price,
                    'entry_idx':i,
                    'hold_period':int(predictions.iloc[i]['holding_period']),
                    'exposure':new_exposure,
                    'peak_pnl_pct':0,
                    'peak_price':current_price
                })
                last_pyramid_idx=i
            
            # 情况2：有持仓，检查加仓
            elif PYRAMID_ENABLED and len(positions)<PYRAMID_MAX_COUNT:
                # 计算当前总盈亏
                total_pnl_pct=0
                for pos in positions:
                    if pos['side']==1:
                        pnl=(current_price-pos['entry_price'])/pos['entry_price']*pos['exposure']
                    else:
                        pnl=(pos['entry_price']-current_price)/pos['entry_price']*pos['exposure']
                    total_pnl_pct+=pnl
                
                # 计算当前总敞口
                current_total_exposure=sum(p['exposure'] for p in positions)
                
                # 加仓条件检查
                can_pyramid=(
                    total_pnl_pct>PYRAMID_PROFIT_THRESHOLD and  # 总体盈利>1%
                    predictions.iloc[i]['direction']==positions[0]['side'] and  # 方向一致
                    predictions.iloc[i]['predicted_rr']>=PYRAMID_MIN_RR and  # 高质量信号
                    predictions.iloc[i]['direction_prob']>=PYRAMID_MIN_PROB and
                    i-last_pyramid_idx>=PYRAMID_MIN_BARS and  # 距上次加仓至少5根K线
                    current_total_exposure+new_exposure<=max_total_exposure  # 总敞口不超限
                )
                
                if can_pyramid:
                    positions.append({
                        'side':predictions.iloc[i]['direction'],
                        'entry_price':current_price,
                        'entry_idx':i,
                        'hold_period':positions[0]['hold_period'],  # 继承首仓周期
                        'exposure':new_exposure,
                        'peak_pnl_pct':0,
                        'peak_price':current_price
                    })
                    last_pyramid_idx=i
                    logger.info(f"[{current_time}]加仓成功! 第{len(positions)}仓,敞口{new_exposure:.1f}倍,总敞口{sum(p['exposure'] for p in positions):.1f}倍")
    
    # 强制平仓
    if len(positions)>0:
        current_price=klines.iloc[-1]['close']
        total_pnl_pct=0
        for pos in positions:
            if pos['side']==1:
                pnl_pct=(current_price-pos['entry_price'])/pos['entry_price']*pos['exposure']
            else:
                pnl_pct=(pos['entry_price']-current_price)/pos['entry_price']*pos['exposure']
            total_pnl_pct+=pnl_pct
        
        total_pnl=equity*total_pnl_pct
        equity+=total_pnl
        
        total_exposure=sum(p['exposure'] for p in positions)
        avg_entry=sum(p['entry_price']*p['exposure'] for p in positions)/total_exposure if total_exposure>0 else positions[0]['entry_price']
        
        trades.append({
            'entry_time':klines.iloc[positions[0]['entry_idx']]['open_time'],
            'exit_time':klines.iloc[-1]['open_time'],
            'side':'long' if positions[0]['side']==1 else 'short',
            'entry_price':avg_entry,
            'exit_price':current_price,
            'exposure':total_exposure,
            'pyramid_count':len(positions),
            'price_change_pct':((current_price-avg_entry)/avg_entry if positions[0]['side']==1 else (avg_entry-current_price)/avg_entry)*100,
            'pnl_pct':total_pnl_pct*100,
            'pnl':total_pnl,
            'equity_after':equity,
            'close_reason':'强制平仓',
            'stop_loss_hit':False,
            'trailing_stop_hit':False,
            'consecutive_losses':consecutive_losses
        })
    
    # 统计
    total_return=(equity/initial_balance-1)*100
    if len(trades)>0:
        trades_df=pd.DataFrame(trades)
        winning_trades=(trades_df['pnl']>0).sum()
        win_rate=winning_trades/len(trades)*100
        avg_win=trades_df[trades_df['pnl']>0]['pnl'].mean() if winning_trades>0 else 0
        avg_loss=abs(trades_df[trades_df['pnl']<=0]['pnl'].mean()) if (trades_df['pnl']<=0).sum()>0 else 0
        profit_loss_ratio=avg_win/avg_loss if avg_loss>0 else 0
        peak=trades_df['equity_after'].expanding().max()
        max_drawdown=(peak-trades_df['equity_after'])/peak*100
        max_drawdown=max_drawdown.max()
        avg_exposure=trades_df['exposure'].mean()
        pyramid_trades=(trades_df['pyramid_count']>1).sum()
        avg_pyramid_exposure=trades_df[trades_df['pyramid_count']>1]['exposure'].mean() if pyramid_trades>0 else 0
    else:
        win_rate,profit_loss_ratio,max_drawdown,avg_exposure,pyramid_trades,avg_pyramid_exposure,trades_df=0,0,0,0,0,0,None
    
    return {
        'total_return':total_return,
        'final_equity':equity,
        'total_trades':len(trades),
        'win_rate':win_rate,
        'profit_loss_ratio':profit_loss_ratio,
        'max_drawdown':max_drawdown,
        'avg_exposure':avg_exposure,
        'pyramid_trades':pyramid_trades,
        'avg_pyramid_exposure':avg_pyramid_exposure,
        'trades':trades_df
    }

def run_backtest():
    """主回测流程"""
    logger.info("="*80)
    logger.info("金字塔加仓回测-2024年模型-复利+动态敞口")
    logger.info("="*80)
    logger.info(f"\n回测时间范围:{BACKTEST_START}至{BACKTEST_END}")
    logger.info(f"加仓配置: 盈利>{PYRAMID_PROFIT_THRESHOLD*100}%触发, 最多{PYRAMID_MAX_COUNT}次, 总敞口≤{MAX_TOTAL_EXPOSURE}倍")
    
    cfg=load_config()
    logger.info("\n加载K线数据...")
    klines_all=load_klines(cfg)
    klines_all['close_time']=pd.to_datetime(klines_all['close_time'])
    
    backtest_start_ts,backtest_end_ts=pd.Timestamp(BACKTEST_START),pd.Timestamp(BACKTEST_END)
    klines_backtest=klines_all[(klines_all['close_time']>=backtest_start_ts)&(klines_all['close_time']<=backtest_end_ts)].reset_index(drop=True)
    logger.info(f"回测集K线数量:{len(klines_backtest)}")
    
    logger.info("\n构建特征...")
    feature_label_data=build_features_and_labels(cfg,klines_backtest)
    X_backtest_full=feature_label_data.features.reset_index(drop=True)
    min_len=min(len(X_backtest_full),len(klines_backtest))
    X_backtest_full=X_backtest_full.iloc[:min_len]
    klines_backtest=klines_backtest.iloc[:min_len].reset_index(drop=True)
    
    logger.info(f"\n加载模型:{MODEL_DIR}")
    strategy=TwoStageRiskRewardStrategy()
    strategy.load(MODEL_DIR)
    
    with open(MODEL_DIR/'top30_features.txt','r') as f:
        top30=[line.strip() for line in f if line.strip()]
    
    X_backtest=X_backtest_full[top30]
    logger.info("\n生成预测信号...")
    predictions=strategy.predict(X_backtest,rr_threshold=RR_THRESHOLD,prob_threshold=PROB_THRESHOLD)
    logger.info(f"应交易样本数:{predictions['should_trade'].sum()}({predictions['should_trade'].sum()/len(predictions)*100:.2f}%)")
    
    logger.info("\n开始回测...")
    results=pyramid_backtest_with_compounding(
        klines=klines_backtest,
        predictions=predictions,
        initial_balance=INITIAL_BALANCE,
        max_total_exposure=MAX_TOTAL_EXPOSURE,
        stop_loss_pct=STOP_LOSS_PCT,
        max_daily_loss_pct=MAX_DAILY_LOSS_PCT,
        max_drawdown_pause=MAX_DRAWDOWN_PAUSE,
        use_trailing_stop=USE_TRAILING_STOP
    )
    
    logger.info("\n"+"="*80)
    logger.info("回测结果")
    logger.info("="*80)
    logger.info(f"总收益率: {results['total_return']:.2f}%")
    logger.info(f"最终权益: {results['final_equity']:,.2f} USDT")
    logger.info(f"总交易数: {results['total_trades']}")
    logger.info(f"胜率: {results['win_rate']:.2f}%")
    logger.info(f"盈亏比: {results['profit_loss_ratio']:.2f}")
    logger.info(f"最大回撤: {results['max_drawdown']:.2f}%")
    logger.info(f"平均敞口: {results['avg_exposure']:.2f}倍")
    logger.info(f"\n加仓统计:")
    logger.info(f"包含加仓的交易数: {results['pyramid_trades']}")
    logger.info(f"加仓交易平均总敞口: {results['avg_pyramid_exposure']:.2f}倍")
    
    if results['trades'] is not None:
        OUTPUT_DIR.mkdir(exist_ok=True)
        output_file=OUTPUT_DIR/f"pyramid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results['trades'].to_csv(output_file,index=False)
        logger.info(f"\n交易明细已保存:{output_file}")
    
    logger.info("\n回测完成!")
    return results

if __name__=="__main__":
    run_backtest()