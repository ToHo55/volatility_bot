import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any
from src.strategies import MLBoostSignal, MeanRevertSignal, BreakoutATRSignal
from src.backtest import BacktestEngine
from src.datasource import DataSource

def optimize_ml_boost(trial: optuna.Trial, df: pd.DataFrame) -> float:
    """ML Boost 전략 파라미터 최적화"""
    # 파라미터 탐색
    threshold = trial.suggest_float('threshold', 0.5, 0.8)
    
    # 전략 실행
    strategy = MLBoostSignal()
    engine = BacktestEngine()
    _, metrics = engine.run_backtest(df.copy(), strategy, 'ml_boost')
    
    # 목표: 샤프 비율 최대화
    return metrics['sharpe_ratio']

def optimize_mean_revert(trial: optuna.Trial, df: pd.DataFrame) -> float:
    """평균회귀 전략 파라미터 최적화"""
    # 파라미터 탐색
    rsi_period = trial.suggest_int('rsi_period', 10, 30)
    bb_period = trial.suggest_int('bb_period', 10, 30)
    bb_std = trial.suggest_float('bb_std', 1.5, 2.5)
    rsi_oversold = trial.suggest_int('rsi_oversold', 20, 30)
    rsi_overbought = trial.suggest_int('rsi_overbought', 60, 80)
    ema_period = trial.suggest_int('ema_period', 10, 30)
    trend_threshold = trial.suggest_float('trend_threshold', 0.1, 0.5)
    
    # 전략 실행
    strategy = MeanRevertSignal(
        rsi_period=rsi_period,
        bb_period=bb_period,
        bb_std=bb_std,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        ema_period=ema_period,
        trend_threshold=trend_threshold
    )
    engine = BacktestEngine()
    _, metrics = engine.run_backtest(df.copy(), strategy, 'mean_revert')
    
    # 목표: 샤프 비율 최대화
    return metrics['sharpe_ratio']

def optimize_breakout_atr(trial: optuna.Trial, df: pd.DataFrame) -> float:
    """돌파 + ATR 전략 파라미터 최적화"""
    # 파라미터 탐색
    atr_period = trial.suggest_int('atr_period', 10, 30)
    entry_multiplier = trial.suggest_float('entry_multiplier', 0.7, 1.3)
    trail_multiplier = trial.suggest_float('trail_multiplier', 1.2, 2.0)
    risk_per_trade = trial.suggest_float('risk_per_trade', 0.01, 0.05)
    
    # 전략 실행
    strategy = BreakoutATRSignal(
        atr_period=atr_period,
        entry_multiplier=entry_multiplier,
        trail_multiplier=trail_multiplier,
        risk_per_trade=risk_per_trade
    )
    engine = BacktestEngine()
    _, metrics = engine.run_backtest(df.copy(), strategy, 'breakout_atr')
    
    # 목표: 샤프 비율 최대화
    return metrics['sharpe_ratio']

if __name__ == "__main__":
    # 데이터 수집
    ds = DataSource()
    df = ds.get_historical_data("KRW-BTC", "1d", "1y")
    
    # 전략별 최적화
    strategies = {
        'ml_boost': optimize_ml_boost,
        'mean_revert': optimize_mean_revert,
        'breakout_atr': optimize_breakout_atr
    }
    
    results = {}
    for name, objective in strategies.items():
        print(f"\nOptimizing {name}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, df), n_trials=200)
        
        print(f"Best trial for {name}:")
        print(f"  Value: {study.best_trial.value}")
        print("  Params: ")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
            
        results[name] = study.best_trial.params 