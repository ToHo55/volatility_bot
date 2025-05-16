import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any

class MeanRevertSignal:
    """RSI + 볼린저 밴드 기반 평균회귀 전략"""
    
    def __init__(self, 
                 rsi_period: int = 14,
                 bb_period: int = 20,
                 bb_std: float = 1.5,
                 rsi_oversold: int = 40,
                 rsi_overbought: int = 60,
                 ema_period: int = 20,
                 trend_threshold: float = 0.1):
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.ema_period = ema_period
        self.trend_threshold = trend_threshold
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        # RSI
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_period)
        
        # 볼린저 밴드
        bb = ta.bbands(df['close'], length=self.bb_period, std=self.bb_std)
        df['bb_upper'] = bb['BBU_20_1.5']
        df['bb_middle'] = bb['BBM_20_1.5']
        df['bb_lower'] = bb['BBL_20_1.5']
        
        # EMA
        df['ema'] = ta.ema(df['close'], length=self.ema_period)
        
        return df
        
    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """매매 신호 생성"""
        # 지표 계산
        df = self.calculate_indicators(df)
        
        # 신호 생성
        signals = pd.Series(0, index=df.index)
        
        # 롱 진입 조건
        long_condition = (
            (df['rsi'] < self.rsi_oversold) &
            (df['close'] < df['bb_lower'])
        )
        
        # 숏 진입 조건
        short_condition = (
            (df['rsi'] > self.rsi_overbought) &
            (df['close'] > df['bb_upper'])
        )
        
        # 청산 조건
        exit_condition = (
            ((df['rsi'] > 50) & (signals.shift(1) == 1)) |
            ((df['rsi'] < 50) & (signals.shift(1) == -1))
        )
        
        # 신호 적용
        signals[long_condition] = 1
        signals[short_condition] = -1
        signals[exit_condition] = 0
        
        return signals 