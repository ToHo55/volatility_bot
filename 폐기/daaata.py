import pandas as pd

# 절대경로로 pkl 파일 불러오기
df = pd.read_pickle(r'C:/Cording/volatility_bot/cache/0f9039bf4dc80a98c0d73ddae8a32180.pkl')

# 2. 인덱스가 datetime이면 reset_index()
if not 'open' in df.columns:
    df = df.reset_index()

# 3. 컬럼명 통일 (필요시)
df = df.rename(columns={'value':'volume', 'candle_acc_trade_volume':'volume'})

# 4. 필요한 컬럼만 추출
df = df[['open', 'high', 'low', 'close', 'volume']]

# 5. (선택) data.csv로 저장
df.to_csv('data.csv', index=False)
print("data.csv로 저장 완료!")