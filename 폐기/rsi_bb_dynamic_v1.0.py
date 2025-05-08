import pandas as pd
import numpy as np
import ta
import os

FEE = 0.0005  # 업비트 0.05%

# 데이터 불러오기 (예시: data.csv)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'data.csv')
df = pd.read_csv(csv_path)
df['datetime'] = pd.to_datetime(df['datetime'])

# 지표 계산
# RSI(14)
df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
# MA(3), MA(10)
df['ma3'] = df['close'].rolling(window=3).mean()
df['ma10'] = df['close'].rolling(window=10).mean()
# 볼린저밴드(20, 2)
bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
df['bb_upper'] = bb.bollinger_hband()
df['bb_lower'] = bb.bollinger_lband()
# 거래량 이동평균
df['vol_ma10'] = df['volume'].rolling(window=10).mean()
df['vol_ma20'] = df['volume'].rolling(window=20).mean()
# 5분봉 변동성 계산
df['volatility'] = df['close'].pct_change().rolling(5).std()

# NaN 제거 (여기서!)
df = df.dropna().reset_index(drop=True)

# 1. 시장상태 판별 함수 (명확하게)
def get_market_state(row):
    rsi = row['rsi']
    ma_short = row['ma3']
    ma_long = row['ma10']
    ma_diff = (ma_short - ma_long) / ma_long if ma_long != 0 else 0
    volatility = row['volatility']
    if rsi > 53 and ma_diff >= 0.001:
        return 'bull'
    elif 43 <= rsi <= 57 and abs(ma_diff) < 0.006:
        return 'sideways'
    elif rsi < 47 and ma_diff <= -0.001:
        return 'bear'
    elif volatility > 0.005:
        return 'volatile'
    else:
        return 'unknown'

df['market_state'] = df.apply(get_market_state, axis=1)

# 2. 1차 진입 패턴 필터
def entry_pattern_filter(row):
    close = row['close']
    upper = row['bb_upper']
    lower = row['bb_lower']
    rsi = row['rsi']
    market_state = row['market_state']
    if market_state == 'bull':
        if close > upper * 1.005 and rsi > 60:
            return True
    elif market_state == 'sideways':
        if close < lower * 1.005 and rsi < 50:
            return True
    elif market_state == 'volatile':
        if close < lower * 1.01 and rsi < 54:
            return True
    return False

df['entry_pattern'] = df.apply(entry_pattern_filter, axis=1)

# 3. 2차 보조 필터 (거래량 + 캔들패턴)
def volume_candle_filter(df, idx):
    if idx < 20:
        return False
    vol = df['volume'].iloc[idx]
    avg10 = df['vol_ma10'].iloc[idx]
    avg20 = df['vol_ma20'].iloc[idx]
    # 거래량 급증
    if not (vol > avg10 * 3 and vol > avg20 * 2):
        return False
    # 캔들패턴: 최근 2봉 중 하나라도 장대 양/음봉 or body > 1%
    for i in range(2):
        o = df['open'].iloc[idx-i]
        c = df['close'].iloc[idx-i]
        h = df['high'].iloc[idx-i]
        l = df['low'].iloc[idx-i]
        body = abs(c - o)
        wick = (h - max(c, o)) + (min(c, o) - l)
        if body > wick or (o != 0 and body / o > 0.01):
            return True
    return False

# 4. TP/SL 설정 함수 (시장상태별)
def get_tp_sl(market_state):
    if market_state == 'bull':
        return 0.015, 0.007
    elif market_state == 'sideways':
        return 0.004, 0.002
    elif market_state == 'volatile':
        return 0.012, 0.005
    else:
        return 0.01, 0.01

# 5. entry_flag: entry_pattern이 True가 된 index의 "다음 봉"에만 True, 2봉 이상 지나면 무효
df['entry_flag'] = False
for idx in range(1, len(df)-1):
    if df['entry_pattern'].iloc[idx-1] == False and df['entry_pattern'].iloc[idx] == True:
        df.at[idx+1, 'entry_flag'] = True

position = 0
entry_price = 0
results = []
trailing_stop = None
max_price = 0
entry_idx = None
entry_flag_idx = None

for idx in range(21, len(df)):
    row = df.iloc[idx]
    prev_row = df.iloc[idx-1]
    # 진입 조건: entry_flag True, 포지션 없음, prev_row['market_state'] in ['bull', 'sideways', 'volatile']
    # prev_row['market_state'] == 'bear' or 'unknown'이면 무조건 진입 금지
    if (position == 0 and row['entry_flag'] and 
        df['entry_pattern'].iloc[idx] and
        prev_row['market_state'] in ['bull', 'sideways', 'volatile']):
        # 2차 보조 필터까지 모두 통과해야 진입
        if volume_candle_filter(df, idx):
            position = 1
            entry_price = row['close'] * (1 + FEE)  # 매수 수수료 반영
            tp, sl = get_tp_sl(prev_row['market_state'])
            trailing_stop = entry_price * (1 - sl)
            max_price = entry_price
            entry_idx = idx
            entry_flag_idx = idx
            continue
    # 진입 신호 후 2봉 이상 지나면 entry_flag 무효
    if position == 0 and entry_flag_idx is not None and idx - entry_flag_idx > 1:
        entry_flag_idx = None
    # 포지션 보유 중
    if position == 1:
        high = row['high']
        low = row['low']
        # 트레일링 스탑: +0.5% 이상 수익 시, 손절가를 진입가로 올림
        if high > entry_price * 1.005:
            trailing_stop = max(trailing_stop, entry_price)
        max_price = max(max_price, high)
        # 익절
        if high >= entry_price * (1 + tp):
            exit_price = entry_price * (1 + tp) * (1 - FEE)  # 매도 수수료 반영
            results.append({'entry': entry_price, 'exit': exit_price, 'type': 'TP', 'entry_idx': entry_idx, 'exit_idx': idx})
            position = 0
            entry_flag_idx = None
        # 트레일링 스탑 or 손절
        elif low <= trailing_stop:
            exit_price = trailing_stop * (1 - FEE)
            results.append({'entry': entry_price, 'exit': exit_price, 'type': 'SL', 'entry_idx': entry_idx, 'exit_idx': idx})
            position = 0
            entry_flag_idx = None

# 결과 요약
profit = sum([r['exit'] - r['entry'] for r in results])
print(f"총 트레이드 수: {len(results)}")
print(f"총 수익: {profit:.4f}")
print(f"TP: {sum(1 for r in results if r['type']=='TP')}, SL: {sum(1 for r in results if r['type']=='SL')}")

print("\n--- 1단계 성공 체크리스트 자동 검증 ---")

# 1. 시장상태별 구간 수
print("\n[1] 시장상태별 구간 수")
print(df['market_state'].value_counts())

# 2. 하락장 진입 여부
bear_entries = [r for r in results if df.iloc[r['entry_idx']-1]['market_state'] == 'bear']
print(f"\n[2] 하락장 진입 건수: {len(bear_entries)} (0이어야 정상)")

# 3. 1차+2차 필터 통과 진입만 존재하는가?
fail_count = 0
for r in results:
    idx = r['entry_idx']
    if not df['entry_pattern'].iloc[idx]:
        print(f"1차 패턴 미통과 진입: {idx}")
        fail_count += 1
    if not volume_candle_filter(df, idx):
        print(f"2차 필터 미통과 진입: {idx}")
        fail_count += 1
if fail_count == 0:
    print("\n[3] 모든 진입이 1차+2차 필터 통과")
else:
    print(f"\n[3] 필터 미통과 진입 {fail_count}건")

# 4. TP/SL 동적 적용 정상 체크
import numpy as np
for r in results:
    entry = r['entry']
    exit = r['exit']
    tp, sl = get_tp_sl(df.iloc[r['entry_idx']-1]['market_state'])
    if r['type'] == 'TP':
        expected = entry * (1 + tp) * (1 - FEE)
        if not np.isclose(exit, expected, atol=1e-2):
            print(f"TP 청산가 이상: {r}")
    if r['type'] == 'SL':
        min_sl = entry * (1 - sl) * (1 - FEE)
        if exit > min_sl + 1e-2:
            print(f"SL 청산가 이상: {r}")
print("\n[4] TP/SL 동적 적용 체크 완료")

# 5. 시장상태별 수익률 차이
from collections import defaultdict
state_profit = defaultdict(list)
for r in results:
    state = df.iloc[r['entry_idx']-1]['market_state']
    state_profit[state].append(r['exit'] - r['entry'])

print("\n[5] 시장상태별 수익률")
for state, profits in state_profit.items():
    if profits:
        print(f"{state}: 진입 {len(profits)}회, 총수익 {sum(profits):.4f}, 평균수익 {np.mean(profits):.4f}, 표준편차 {np.std(profits):.4f}")
    else:
        print(f"{state}: 진입 없음")

# 트레이드별 퍼센트 수익률
percent_profits = [(r['exit'] - r['entry']) / r['entry'] for r in results]

# 누적 수익률(복리, %) 시계열
cum_returns = [1]
for p in percent_profits:
    cum_returns.append(cum_returns[-1] * (1 + p))
cum_returns = np.array(cum_returns)
cum_return_percent = (cum_returns[-1] - 1) * 100

print(f"\n누적 수익률(복리): {cum_return_percent:.2f}%")

# 최대 낙폭(MDD, %)
peak = np.maximum.accumulate(cum_returns)
drawdown = (cum_returns - peak) / peak
mdd = drawdown.min() * 100  # %
print(f"최대 낙폭(MDD): {mdd:.2f}%")

# 샤프비율 (트레이드 기준, %)
mean_return = np.mean(percent_profits)
std_return = np.std(percent_profits)
if std_return > 0:
    sharpe = mean_return / std_return * np.sqrt(len(percent_profits))
    print(f"샤프비율(트레이드 기준): {sharpe:.2f}")
else:
    print("샤프비율: 계산 불가(표준편차=0)")

# 트레이드당 평균 수익률(%)도 함께 출력
print(f"평균 트레이드당 수익률: {mean_return*100:.4f}%")

from collections import defaultdict
year_profit = defaultdict(list)
for r in results:
    dt = df.iloc[r['entry_idx']]['datetime']
    if dt is not None:
        year = dt.year
        year_profit[year].append((r['exit'] - r['entry']) / r['entry'] * 100)
for year, profits in year_profit.items():
    if profits:
        print(f"{year}: 진입 {len(profits)}회, 총수익률 {sum(profits):.2f}%, 평균수익률 {np.mean(profits):.2f}%")

# 실험(백테스트) 전체 기간 출력
if 'datetime' in df.columns:
    print(f"\n실험 기간: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
elif 'timestamp' in df.columns:
    print(f"\n실험 기간: {pd.to_datetime(df['timestamp'].iloc[0])} ~ {pd.to_datetime(df['timestamp'].iloc[-1])}")
else:
    print(f"\n데이터 인덱스(행 번호): {df.index[0]} ~ {df.index[-1]}")
