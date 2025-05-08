import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Union, Optional
from loguru import logger

class TechnicalIndicators:
    def __init__(self):
        """기술적 지표 계산 클래스 초기화"""
        pass

    def calc_rsi(self, series: pd.Series, n: int = 14) -> pd.Series:
        """
        RSI(Relative Strength Index) 계산
        
        Args:
            series (pd.Series): 가격 데이터
            n (int): 기간 (기본값: 14)
            
        Returns:
            pd.Series: RSI 값 (0-100 사이로 클리핑)
        """
        try:
            rsi = ta.rsi(series, length=n)
            return rsi.clip(0, 100)  # 0-100 사이로 클리핑
        except Exception as e:
            logger.error(f"RSI 계산 중 오류 발생: {e}")
            return pd.Series(index=series.index)

    def calc_ema(self, series: pd.Series, n: int) -> pd.Series:
        """
        EMA(Exponential Moving Average) 계산
        
        Args:
            series (pd.Series): 가격 데이터
            n (int): 기간
            
        Returns:
            pd.Series: EMA 값
        """
        try:
            return ta.ema(series, length=n)
        except Exception as e:
            logger.error(f"EMA 계산 중 오류 발생: {e}")
            return pd.Series(index=series.index)

    def ema_slope(self, series: pd.Series, n: int = 5) -> pd.Series:
        """
        EMA 기울기 계산 (Δ/Δt)
        
        Args:
            series (pd.Series): 가격 데이터
            n (int): 기간 (기본값: 5)
            
        Returns:
            pd.Series: EMA 기울기 (양수/음수)
        """
        try:
            ema = self.calc_ema(series, n)
            # 기울기 계산 (현재값 - 이전값)
            slope = ema.diff()
            return slope > 0  # True: 상승, False: 하락
        except Exception as e:
            logger.error(f"EMA 기울기 계산 중 오류 발생: {e}")
            return pd.Series(index=series.index)

    def calc_atr(self, high: pd.Series, low: pd.Series, 
                close: pd.Series, n: int = 14) -> pd.Series:
        """
        ATR(Average True Range) 계산
        
        Args:
            high (pd.Series): 고가 데이터
            low (pd.Series): 저가 데이터
            close (pd.Series): 종가 데이터
            n (int): 기간 (기본값: 14)
            
        Returns:
            pd.Series: ATR 값
        """
        try:
            return ta.atr(high=high, low=low, close=close, length=n)
        except Exception as e:
            logger.error(f"ATR 계산 중 오류 발생: {e}")
            return pd.Series(index=close.index)

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터프레임에 모든 기술적 지표 추가
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            
        Returns:
            pd.DataFrame: 지표가 추가된 데이터프레임
        """
        try:
            # RSI
            df['rsi'] = self.calc_rsi(df['close'])
            
            # EMA
            df['ema5'] = self.calc_ema(df['close'], 5)
            df['ema20'] = self.calc_ema(df['close'], 20)
            
            # EMA 기울기
            df['ema5_slope'] = self.ema_slope(df['close'], 5)
            
            # ATR
            df['atr'] = self.calc_atr(df['high'], df['low'], df['close'])
            
            return df
        except Exception as e:
            logger.error(f"지표 추가 중 오류 발생: {e}")
            return df

if __name__ == "__main__":
    # 테스트 코드
    import yfinance as yf
    
    # 테스트 데이터 다운로드
    data = yf.download("BTC-USD", start="2024-01-01", end="2024-05-08", interval="1h")
    
    # 지표 계산
    ti = TechnicalIndicators()
    df = ti.add_indicators(data)
    
    print("계산된 지표:")
    print(df[['rsi', 'ema5', 'ema20', 'ema5_slope', 'atr']].head()) 