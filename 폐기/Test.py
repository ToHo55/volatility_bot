import pandas as pd
import ta
import matplotlib.pyplot as plt

# 5분봉 데이터 (피클) 불러오기
print('5분봉 데이터 로딩...')
df_5min = pd.read_pickle('C:/Cording/volatility_bot/0f9039bf4dc80a98c0d73ddae8a32180.pkl')
if isinstance(df_5min.index, pd.DatetimeIndex):
    df_5min = df_5min.reset_index()
    df_5min = df_5min.rename(columns={'index': 'datetime'})
df_5min['datetime'] = pd.to_datetime(df_5min['datetime'])

# 숫자 컬럼 float 변환 (혹시 모를 오류 방지)
for col in ['open', 'high', 'low', 'close', 'volume']:
    df_5min[col] = pd.to_numeric(df_5min[col], errors='coerce')

# 지표 계산 함수
def add_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['ma3'] = df['close'].rolling(window=3).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['volatility'] = df['close'].pct_change().rolling(5).std()
    return df.dropna().reset_index(drop=True)

df_5min = add_indicators(df_5min)
df_5min['ma10_prev'] = df_5min['ma10'].shift(1)
df_5min = df_5min.dropna().reset_index(drop=True)

# 시장상태 분류 함수
def get_market_state_5min(row):
    bb_width = (row['bb_upper'] - row['bb_lower']) / row['close'] if row['close'] != 0 else 0
    if row['close'] > row['ma10'] and row['ma3'] > row['ma10'] and row['ma10'] > row['ma10_prev']:
        return 'bull'
    elif row['close'] < row['ma10'] and row['ma3'] < row['ma10'] and row['ma10'] < row['ma10_prev']:
        return 'bear'
    elif abs((row['ma3'] - row['ma10']) / row['ma10']) < 0.003 and row['volatility'] < 0.001 and bb_width < 0.01:
        return 'box'
    elif abs((row['ma3'] - row['ma10']) / row['ma10']) < 0.007 and row['volatility'] < 0.002 and bb_width < 0.015:
        return 'sideways'
    elif row['volatility'] < 0.0005:
        return 'super_low_vol'
    else:
        return 'unknown'

df_5min['market_state'] = df_5min.apply(get_market_state_5min, axis=1)

# 분포 출력
print('\n[5분봉 시장상태 분포]')
print(df_5min['market_state'].value_counts())
print(df_5min['market_state'].value_counts(normalize=True))

# 분석 구간 설정
start_date = pd.Timestamp('2023-04-28')
end_date = pd.Timestamp('2025-04-28')

# 5분봉 최근 1년 분포
df_5min_period = df_5min[(df_5min['datetime'] >= start_date) & (df_5min['datetime'] < end_date)]
print('\n[5분봉 시장상태 분포 - 최근 3년]')
print(df_5min_period['market_state'].value_counts())
print(df_5min_period['market_state'].value_counts(normalize=True))

# 시각화 구간 설정 (2일)
plot_start = pd.Timestamp('2025-04-22')
plot_end = pd.Timestamp('2025-04-24')

# 상태별 색상 매핑
def get_state_color():
    return {
        'bull': 'red',
        'bear': 'blue',
        'box': 'green',
        'sideways': 'orange',
        'super_low_vol': 'gray',
        'unknown': 'black'
    }

# 5분봉 시각화
df_plot_5min = df_5min[(df_5min['datetime'] >= plot_start) & (df_5min['datetime'] < plot_end)].copy()
colors_5min = df_plot_5min['market_state'].map(get_state_color())
plt.figure(figsize=(18, 6))
plt.scatter(df_plot_5min['datetime'], df_plot_5min['close'], c=colors_5min, s=5)
plt.title('5분봉 차트 + market_state 색상 (2025-04-22 ~ 2025-04-24)')
plt.xlabel('datetime')
plt.ylabel('close')
plt.tight_layout()
plt.show()

print(f"5분봉 {plot_start} ~ {plot_end} 데이터 개수:", len(df_plot_5min))
print(df_plot_5min[['datetime', 'close', 'market_state']].head())
print(df_plot_5min[['datetime', 'close', 'market_state']].tail())
print("5분봉 데이터 전체 범위:", df_5min['datetime'].min(), "~", df_5min['datetime'].max())