import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from datetime import datetime, timedelta
from .strategies import MLBoostSignal, MeanRevertSignal, BreakoutATRSignal

class BacktestEngine:
    """백테스팅 엔진"""
    
    def __init__(self,
                 initial_capital: float = 1000000,
                 slippage: float = 0.0005,  # 0.05%
                 fee: float = 0.0005,       # 0.05%
                 risk_free_rate: float = 0.02):  # 연 2%
        self.initial_capital = initial_capital
        self.slippage = slippage
        self.fee = fee
        self.risk_free_rate = risk_free_rate
        
    def calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """성과 지표 계산"""
        # 연간화된 수익률
        annual_return = (1 + returns.mean()) ** 252 - 1
        
        # 샤프 비율 (분모가 0일 때 예외처리)
        excess_returns = returns - self.risk_free_rate/252
        std = returns.std()
        if std == 0 or np.isnan(std):
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / std
        
        # 최대 낙폭
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # 월간 거래 횟수
        trades_per_month = len(returns[returns != 0]) / (len(returns) / 21)
        
        return {
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades_per_month': trades_per_month
        }
        
    def run_backtest(self,
                    df: pd.DataFrame,
                    strategy: Any,
                    strategy_name: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """백테스팅 실행"""
        # 신호 생성
        signals = strategy.generate_signal(df)
        position_sizes = pd.Series(1.0, index=df.index)
        
        # 신호 발생 개수 및 데이터 head/tail 출력
        print(f"[DEBUG] {strategy_name} 신호 발생 개수: {signals.value_counts().to_dict()}")
        print(f"[DEBUG] {strategy_name} 데이터 head:\n{df.head()}")
        print(f"[DEBUG] {strategy_name} 데이터 tail:\n{df.tail()}")
        
        # 포지션 초기화
        df['position'] = 0
        df['holdings'] = 0
        df['cash'] = self.initial_capital
        
        # 포지션 진입/청산
        for i in range(1, len(df)):
            if signals.iloc[i] != 0 and df['position'].iloc[i-1] == 0:
                # 진입
                price = df['close'].iloc[i] * (1 + self.slippage if signals.iloc[i] > 0 else 1 - self.slippage)
                size = position_sizes.iloc[i]
                cost = price * size * (1 + self.fee)
                
                if cost <= df['cash'].iloc[i-1]:
                    df.loc[df.index[i], 'position'] = signals.iloc[i]
                    df.loc[df.index[i], 'holdings'] = size
                    df.loc[df.index[i], 'cash'] = df['cash'].iloc[i-1] - cost
                    print(f"[TRADE] {strategy_name} 진입: {df.index[i]}, 가격: {price}, 포지션: {signals.iloc[i]}")
                else:
                    df.loc[df.index[i], 'position'] = 0
                    df.loc[df.index[i], 'holdings'] = 0
                    df.loc[df.index[i], 'cash'] = df['cash'].iloc[i-1]
                    
            elif signals.iloc[i] == 0 and df['position'].iloc[i-1] != 0:
                # 청산
                price = df['close'].iloc[i] * (1 - self.slippage if df['position'].iloc[i-1] > 0 else 1 + self.slippage)
                size = df['holdings'].iloc[i-1]
                proceeds = price * size * (1 - self.fee)
                
                df.loc[df.index[i], 'position'] = 0
                df.loc[df.index[i], 'holdings'] = 0
                df.loc[df.index[i], 'cash'] = df['cash'].iloc[i-1] + proceeds
                print(f"[TRADE] {strategy_name} 청산: {df.index[i]}, 가격: {price}, 수익률: {(proceeds / (df['cash'].iloc[i-1] - df['cash'].iloc[i])) - 1:.2%}, 포지션 유지 기간: {i - df.index.get_loc(df.index[i-1])}일")
                
            else:
                # 포지션 유지
                df.loc[df.index[i], 'position'] = df['position'].iloc[i-1]
                df.loc[df.index[i], 'holdings'] = df['holdings'].iloc[i-1]
                df.loc[df.index[i], 'cash'] = df['cash'].iloc[i-1]
        
        # 자산 계산
        df['total_assets'] = df['cash'] + df['holdings'] * df['close']
        df['returns'] = df['total_assets'].pct_change()
        
        # 성과 지표 계산
        metrics = self.calculate_metrics(df['returns'].dropna())
        
        return df, metrics

if __name__ == "__main__":
    # 테스트 코드
    from src.datasource import DataSource
    
    # 데이터 수집
    ds = DataSource()
    df = ds.get_historical_data("KRW-BTC", "1d", "1y")
    
    # 백테스팅 엔진 초기화
    engine = BacktestEngine()
    
    # 전략별 백테스팅 실행
    strategies = {
        'ml_boost': MLBoostSignal(),
        'mean_revert': MeanRevertSignal(),
        'breakout_atr': BreakoutATRSignal()
    }
    
    results = {}
    for name, strategy in strategies.items():
        print(f"\nRunning backtest for {name}...")
        df_result, metrics = engine.run_backtest(df.copy(), strategy, name)
        results[name] = metrics
        print(f"Metrics: {metrics}") 