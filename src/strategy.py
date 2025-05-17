import pandas as pd
import numpy as np

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI 계산"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """이동평균선(MA) 계산"""
    df['MA'] = df['close'].rolling(window=period).mean()
    return df

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ATR(Average True Range) 계산"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=period).mean()
    return df

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """매수/매도 시그널 생성"""
    # RSI, MA, ATR 계산
    df = calculate_rsi(df)
    df = calculate_ma(df)
    df = calculate_atr(df)
    
    # 매수 조건: RSI < 30 (과매도) & 현재가 > MA (상승 추세)
    df['buy_signal'] = (df['RSI'] < 30) & (df['close'] > df['MA'])
    
    # 매도 조건: RSI > 70 (과매수) & 현재가 < MA (하락 추세)
    df['sell_signal'] = (df['RSI'] > 70) & (df['close'] < df['MA'])
    
    return df

if __name__ == "__main__":
    # 테스트 코드
    from src.datasource import DataSource
    ds = DataSource()
    df = ds.get_historical_data("KRW-BTC", "1d", "1y")
    df = generate_signals(df)
    print(df[['timestamp', 'close', 'RSI', 'MA', 'ATR', 'buy_signal', 'sell_signal']].tail()) 