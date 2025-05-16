import pandas as pd
import numpy as np
from typing import Tuple, Optional
from loguru import logger
from .indicators import TechnicalIndicators

class SignalGenerator:
    def __init__(self, rsi_oversold: int = 30, rsi_overbought: int = 55,
                 stop_loss_pct: float = 0.8):
        """
        매매 신호 생성기 초기화
        
        Args:
            rsi_oversold (int): RSI 과매도 기준값
            rsi_overbought (int): RSI 과매수 기준값
            stop_loss_pct (float): 손절 비율 (%)
        """
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.stop_loss_pct = stop_loss_pct
        self.indicators = TechnicalIndicators()
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """매매 시그널 생성"""
        try:
            df = df.copy()
            
            # 포지션 초기화
            df['position'] = 0
            df['stop_price'] = np.nan
            df['stop_loss'] = np.nan
            df['stop_hit'] = False
            
            # 진입 조건 확인
            long_condition = (
                (df['rsi'] < 30) &  # RSI 과매도
                (df['ema5'] > df['ema20']) &  # EMA 골든크로스
                (df['ema5_slope'])  # EMA 상승
            )
            
            # 포지션 진입
            df.loc[long_condition, 'position'] = 1
            
            # 스탑 가격 계산
            for i in range(len(df)):
                if df['position'].iloc[i] == 1:
                    stop_price = self._calculate_stop_price(df.iloc[:i+1])
                    df.loc[df.index[i], 'stop_price'] = stop_price
                    df.loc[df.index[i], 'stop_loss'] = stop_price * 0.99  # 1% 하락 시 손절
            
            # 스탑 히트 확인
            df['stop_hit'] = (df['position'] == 1) & (df['low'] <= df['stop_price'])
            
            return df
        except Exception as e:
            self.logger.error(f"시그널 생성 중 오류 발생: {str(e)}")
            return df
    
    def _calculate_stop_price(self, df: pd.DataFrame) -> float:
        """스탑 가격 계산"""
        try:
            if len(df) < 2:
                return df['close'].iloc[-1] * 0.95  # 기본 5% 하락
            
            # ATR 기반 스탑 가격 계산
            atr = df['atr'].iloc[-1]
            if pd.isna(atr):
                return df['close'].iloc[-1] * 0.95
            
            return df['close'].iloc[-1] - (atr * 2)  # ATR의 2배만큼 하락
        except Exception as e:
            self.logger.error(f"스탑 가격 계산 중 오류 발생: {str(e)}")
            return df['close'].iloc[-1] * 0.95
    
    def _check_stop_loss(self, df: pd.DataFrame) -> pd.Series:
        """
        손절 조건 확인
        
        Args:
            df (pd.DataFrame): OHLCV + 지표 + 손절가 데이터
            
        Returns:
            pd.Series: 손절 여부 (True/False)
        """
        try:
            # 롱 포지션의 경우 저가가 손절가 이하로 떨어지면 손절
            stop_hit = pd.Series(False, index=df.index)
            stop_hit.loc[(df['position'] == 1) & 
                        (df['low'] <= df['stop_price'])] = True
            return stop_hit
            
        except Exception as e:
            logger.error(f"손절 조건 확인 중 오류 발생: {e}")
            return pd.Series(False, index=df.index)

if __name__ == "__main__":
    # 테스트 코드
    import yfinance as yf
    
    # 테스트 데이터 다운로드
    data = yf.download("BTC-USD", start="2024-01-01", end="2024-05-08", interval="1h")
    
    # 신호 생성
    sg = SignalGenerator()
    df = sg.generate_signals(data)
    
    print("생성된 신호:")
    print(df[['position', 'stop_price', 'stop_loss', 'stop_hit']].head())
    
    # 포지션 통계
    total_signals = len(df[df['position'] != 0])
    stop_loss_hits = len(df[df['stop_hit']])
    
    print(f"\n포지션 통계:")
    print(f"총 신호 수: {total_signals}")
    print(f"손절 발생 수: {stop_loss_hits}")
    print(f"손절 비율: {stop_loss_hits/total_signals*100:.2f}%") 