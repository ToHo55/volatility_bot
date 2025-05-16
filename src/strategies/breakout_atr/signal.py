import pandas as pd
import numpy as np
import pandas_ta as ta

class BreakoutATRSignal:
    """ATR 기반 돌파 전략"""
    
    def __init__(self,
                 atr_period: int = 14,
                 atr_multiplier: float = 0.5,
                 min_atr: float = 0.1,
                 max_atr: float = 5.0,
                 stop_loss_atr: float = 0.5,
                 take_profit_atr: float = 1.0):
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.min_atr = min_atr
        self.max_atr = max_atr
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        
    def calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR 계산"""
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
        return df
        
    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """매매 신호 생성"""
        df = self.calculate_atr(df)
        signals = pd.Series(0, index=df.index)
        
        # ATR 필터링
        atr_filter = (df['atr'] >= self.min_atr) & (df['atr'] <= self.max_atr)
        
        # 롱 진입 조건
        long_condition = (
            atr_filter &
            (df['close'] > df['close'].shift(1) * (1 + self.atr_multiplier * df['atr'] / df['close'].shift(1)))
        )
        
        # 숏 진입 조건
        short_condition = (
            atr_filter &
            (df['close'] < df['close'].shift(1) * (1 - self.atr_multiplier * df['atr'] / df['close'].shift(1)))
        )
        
        # 청산 조건
        exit_condition = (
            ((df['close'] < df['close'].shift(1) * (1 - self.stop_loss_atr * df['atr'] / df['close'].shift(1))) & (signals.shift(1) == 1)) |
            ((df['close'] > df['close'].shift(1) * (1 + self.stop_loss_atr * df['atr'] / df['close'].shift(1))) & (signals.shift(1) == -1)) |
            ((df['close'] > df['close'].shift(1) * (1 + self.take_profit_atr * df['atr'] / df['close'].shift(1))) & (signals.shift(1) == 1)) |
            ((df['close'] < df['close'].shift(1) * (1 - self.take_profit_atr * df['atr'] / df['close'].shift(1))) & (signals.shift(1) == -1))
        )
        
        signals[long_condition] = 1
        signals[short_condition] = -1
        signals[exit_condition] = 0
        
        return signals 