import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Union, Optional
from loguru import logger

class TechnicalIndicators:
    def __init__(self):
        """기술적 지표 계산 클래스 초기화"""
        pass

    def calc_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI(Relative Strength Index) 계산
        
        Args:
            series (pd.Series): 가격 데이터
            period (int): 기간 (기본값: 14)
            
        Returns:
            pd.Series: RSI 값 (0-100 사이로 클리핑)
        """
        if period <= 0:
            raise ValueError("RSI 기간은 0보다 커야 합니다.")
            
        if series is None or len(series) == 0:
            raise ValueError("가격 데이터가 비어있습니다.")
            
        if len(series) < period + 1:
            # 데이터가 부족한 경우 NaN으로 채움
            return pd.Series([np.nan] * len(series), index=series.index)
            
        try:
            # 가격 변화 계산
            delta = series.diff()
            
            # 상승/하락 구분
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # 평균 상승/하락 계산
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # RS 계산
            rs = avg_gain / avg_loss
            
            # RSI 계산
            rsi = 100 - (100 / (1 + rs))
            
            # NaN 처리
            rsi = rsi.fillna(50)  # 초기값을 50으로 설정
            
            # 0-100 사이로 클리핑
            rsi = rsi.clip(0, 100)
            
            return rsi
            
        except Exception as e:
            logger.error(f"RSI 계산 중 오류 발생: {e}")
            raise

    def calc_ema(self, series: pd.Series, period: int) -> pd.Series:
        """
        EMA(Exponential Moving Average) 계산
        
        Args:
            series (pd.Series): 가격 데이터
            period (int): 기간
            
        Returns:
            pd.Series: EMA 값
        """
        if period <= 0:
            raise ValueError("EMA 기간은 0보다 커야 합니다.")
            
        if series is None or len(series) == 0:
            raise ValueError("가격 데이터가 비어있습니다.")
            
        try:
            ema = ta.ema(series, length=period)
            ema.name = f'ema{period}'
            return ema
        except Exception as e:
            logger.error(f"EMA 계산 중 오류 발생: {e}")
            raise

    def ema_slope(self, series: pd.Series, period: int = 5) -> pd.Series:
        """
        EMA 기울기 계산 (Δ/Δt)
        
        Args:
            series (pd.Series): 가격 데이터
            period (int): 기간 (기본값: 5)
            
        Returns:
            pd.Series: EMA 기울기 (양수/음수)
        """
        if period <= 0:
            raise ValueError("EMA 기울기 기간은 0보다 커야 합니다.")
            
        if series is None or len(series) == 0:
            raise ValueError("가격 데이터가 비어있습니다.")
            
        try:
            ema = self.calc_ema(series, period)
            # 기울기 계산 (현재값 - 이전값)
            slope = ema.diff()
            slope.name = f'ema{period}_slope'
            return slope > 0  # True: 상승, False: 하락
        except Exception as e:
            logger.error(f"EMA 기울기 계산 중 오류 발생: {e}")
            raise

    def calc_atr(self, high: pd.Series, low: pd.Series, 
                close: pd.Series, period: int = 14) -> pd.Series:
        """
        ATR(Average True Range) 계산
        
        Args:
            high (pd.Series): 고가 데이터
            low (pd.Series): 저가 데이터
            close (pd.Series): 종가 데이터
            period (int): 기간 (기본값: 14)
            
        Returns:
            pd.Series: ATR 값
        """
        if period <= 0:
            raise ValueError("ATR 기간은 0보다 커야 합니다.")
            
        if any(x is None or len(x) == 0 for x in [high, low, close]):
            raise ValueError("가격 데이터가 비어있습니다.")
            
        try:
            atr = ta.atr(high=high, low=low, close=close, length=period)
            atr.name = 'atr'
            return atr
        except Exception as e:
            logger.error(f"ATR 계산 중 오류 발생: {e}")
            raise

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터프레임에 모든 기술적 지표 추가
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            
        Returns:
            pd.DataFrame: 지표가 추가된 데이터프레임
        """
        if df is None or len(df) == 0:
            raise ValueError("데이터프레임이 비어있습니다.")
            
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"필수 컬럼이 누락되었습니다: {required_columns}")
            
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
            raise

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