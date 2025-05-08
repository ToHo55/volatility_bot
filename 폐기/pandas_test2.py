import pandas as pd

def detect_market_condition(df):
    """
    5분봉 데이터프레임(df)에 시장상태(market_condition) 열을 추가하는 함수
    columns: open, high, low, close, volume 필요
    """
    df = df.copy()

    # === 이동평균선 50봉 계산 ===
    df['sma50'] = df['close'].rolling(window=50, min_periods=1).mean()

    # === SMA50 기울기 계산 (현재 - 10봉 전) ===
    df['sma50_slope'] = df['sma50'] - df['sma50'].shift(10)

    # === ATR14 계산 (변동성) ===
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = (df['high'] - df['close'].shift()).abs()
    df['low_close'] = (df['low'] - df['close'].shift()).abs()
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr14'] = df['true_range'].rolling(window=14, min_periods=1).mean()

    # === ATR14의 50봉 이동평균 (장기 변동성 기준) ===
    df['atr14_ma50'] = df['atr14'].rolling(window=50, min_periods=1).mean()

    # === RSI14 계산 ===
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df['rsi14'] = 100 - (100 / (1 + rs))

    # === 시장상태 판별 ===
    conditions = []

    for idx in df.index:
        slope = df.at[idx, 'sma50_slope']
        atr = df.at[idx, 'atr14']
        atr_ma50 = df.at[idx, 'atr14_ma50']
        rsi = df.at[idx, 'rsi14']

        if slope > 0.2 and rsi > 55:
            condition = "STRONG_UPTREND"  # 강한 상승추세
        elif slope < -0.2 and rsi < 45:
            condition = "STRONG_DOWNTREND"  # 강한 하락추세
        else:
            if atr > atr_ma50 * 1.2:
                condition = "SIDEWAYS_VOLATILE"  # 변동성 큰 횡보
            else:
                condition = "SIDEWAYS"  # 일반적 횡보

        conditions.append(condition)

    df['market_condition'] = conditions

    # === 불필요한 중간열 삭제 ===
    df = df.drop(columns=['high_low', 'high_close', 'low_close', 'true_range', 'atr14_ma50'])

    return df
