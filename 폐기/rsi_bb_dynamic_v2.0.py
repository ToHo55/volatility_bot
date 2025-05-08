import pandas as pd
import numpy as np
import ta
import random

FEE = 0.0005  # 업비트 0.05%

# 데이터 불러오기 (예시: data.csv)
df = pd.read_csv('C:/Cording/volatility_bot/data.csv')
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

# === 아래 기존 전체 실험 print/log 코드 완전히 주석 처리 또는 삭제 ===
# print(f"총 트레이드 수: {len(results)}")
# print(f"총 수익: {profit:.4f}")
# print(f"TP: {sum(1 for r in results if r['type']=='TP')}, SL: {sum(1 for r in results if r['type']=='SL')}")
# print("\n--- 1단계 성공 체크리스트 자동 검증 ---")
# print("\n[1] 시장상태별 구간 수")
# print(df['market_state'].value_counts())
# print(f"\n[2] 하락장 진입 건수: {len(bear_entries)} (0이어야 정상)")
# ... (이하 전체 실험 print문 주석 처리) ...
# print("\n--- 2단계 성공 체크리스트 자동 검증 ---")
# print("\n[1] 시장상태별 구간 수")
# print(df['market_state'].value_counts())
# print(f"\n[2] 하락장 진입 건수: {len(bear_entries)} (0이어야 정상)")
# ... (이하 전체 실험 print문 주석 처리) ...

# === 2단계 알고리즘용 함수 정의 ===

def market_state_filter(row):
    # 1차: 시장상태 + bull일 때 ma_diff 추가 필터
    if row['market_state'] not in ['bull', 'sideways', 'volatile']:
        return False
    if row['market_state'] == 'bull':
        ma_short = row['ma3']
        ma_long = row['ma10']
        ma_diff = (ma_short - ma_long) / ma_long if ma_long != 0 else 0
        if ma_diff < 0.003:
            return False
    return True

def pattern_filter(row, idx, df):
    # 2차: 볼린저+RSI+캔들패턴
    close = row['close']
    upper = row['bb_upper']
    lower = row['bb_lower']
    rsi = row['rsi']
    state = row['market_state']
    # 볼린저+RSI
    if state == 'bull':
        if not (close > upper * 1.02 and rsi > 65):
            return False
    elif state == 'sideways':
        if not (close < lower * 1.02 and rsi < 45):
            return False
    elif state == 'volatile':
        if not (close < lower * 1.05 and rsi < 50):
            return False
    # 캔들패턴: 최근 3봉 중 2개 이상 양봉
    if idx < 2:
        return False
    count = 0
    for i in range(3):
        if df['close'].iloc[idx-i] > df['open'].iloc[idx-i]:
            count += 1
    if count < 2:
        return False
    return True

def auxiliary_filter(df, idx):
    # 3차: 거래량 급증 + RSI 흐름
    if idx < 30:
        return False
    v = df['volume'].iloc[idx]
    if not (v > df['vol_ma10'].iloc[idx]*3 and v > df['vol_ma20'].iloc[idx]*2 and v > df['volume'].rolling(30).mean().iloc[idx]*1.5):
        return False
    # RSI 2봉 연속 상승
    if not (df['rsi'].iloc[idx-2] < df['rsi'].iloc[idx-1] < df['rsi'].iloc[idx]):
        return False
    return True

def timing_filter(df, idx):
    # 4차: 신호 다음 봉의 open > 신호봉 close
    if idx+1 >= len(df):
        return False
    return df['open'].iloc[idx+1] > df['close'].iloc[idx]

def get_tp_sl_v2(market_state):
    # 2단계 TP/SL
    if market_state == 'bull':
        return 0.03, 0.015
    elif market_state == 'sideways':
        return 0.007, 0.004
    elif market_state == 'volatile':
        return 0.02, 0.01
    else:
        return 0.01, 0.01

# === 진입 신호 생성 ===
df['entry_signal'] = False
for idx in range(30, len(df)-1):
    row = df.iloc[idx]
    if not market_state_filter(row):
        continue
    if not pattern_filter(row, idx, df):
        continue
    if not auxiliary_filter(df, idx):
        continue
    if not timing_filter(df, idx):
        continue
    df.at[idx+1, 'entry_signal'] = True  # 신호 다음 봉에 진입

# === 포지션 관리 및 쿨다운 ===
position = 0
entry_price = 0
results = []
trailing_stop = None
max_price = 0
entry_idx = None
cooldown = 0
loss_streak = 0

for idx in range(31, len(df)):
    row = df.iloc[idx]
    prev_row = df.iloc[idx-1]
    # 쿨다운 중이면 진입 금지
    if cooldown > 0:
        cooldown -= 1
    # 진입 조건: entry_signal True, 포지션 없음, 쿨다운 아님
    if position == 0 and row['entry_signal'] and cooldown == 0:
        position = 1
        entry_price = row['close'] * (1 + FEE)
        tp, sl = get_tp_sl_v2(prev_row['market_state'])
        trailing_stop = entry_price * (1 - sl)
        max_price = entry_price
        entry_idx = idx
        continue
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
            exit_price = entry_price * (1 + tp) * (1 - FEE)
            results.append({'entry': entry_price, 'exit': exit_price, 'type': 'TP', 'entry_idx': entry_idx, 'exit_idx': idx})
            position = 0
            entry_idx = None
            loss_streak = 0
        # 트레일링 스탑 or 손절
        elif low <= trailing_stop:
            exit_price = trailing_stop * (1 - FEE)
            results.append({'entry': entry_price, 'exit': exit_price, 'type': 'SL', 'entry_idx': entry_idx, 'exit_idx': idx})
            position = 0
            entry_idx = None
            loss_streak += 1
            if loss_streak >= 3:
                cooldown = 10
        # 포지션 종료 후 쿨다운 적용
        if position == 0 and loss_streak >= 3:
            cooldown = 10

# === 아래 기존 전체 실험 print/log 코드 완전히 주석 처리 또는 삭제 ===
# print(f"총 트레이드 수: {len(results)}")
# print(f"총 수익: {profit:.4f}")
# print(f"TP: {sum(1 for r in results if r['type']=='TP')}, SL: {sum(1 for r in results if r['type']=='SL')}")
# print("\n--- 2단계 성공 체크리스트 자동 검증 ---")
# print("\n[1] 시장상태별 구간 수")
# print(df['market_state'].value_counts())
# print(f"\n[2] 하락장 진입 건수: {len(bear_entries)} (0이어야 정상)")
# ... (이하 전체 실험 print문 주석 처리) ...
# print("\n[3] 1차+2차 필터 통과 진입만 존재하는가?")
# ... (이하 전체 실험 print문 주석 처리) ...
# print("\n[4] TP/SL 동적 적용 정상 체크")
# ... (이하 전체 실험 print문 주석 처리) ...
# print("\n[5] 시장상태별 수익률 차이")
# ... (이하 전체 실험 print문 주석 처리) ...
# print(f"\n누적 수익률(복리): {cum_return_percent:.2f}%")
# ... (이하 전체 실험 print문 주석 처리) ...
# print(f"최대 낙폭(MDD): {mdd:.2f}%")
# ... (이하 전체 실험 print문 주석 처리) ...
# print(f"샤프비율(트레이드 기준): {sharpe:.2f}")
# ... (이하 전체 실험 print문 주석 처리) ...
# print(f"평균 트레이드당 수익률: {mean_return*100:.4f}%")
# ... (이하 전체 실험 print문 주석 처리) ...
# from collections import defaultdict
# year_profit = defaultdict(list)
# for r in results:
#     dt = df.iloc[r['entry_idx']]['datetime']
#     if dt is not None:
#         year = dt.year
#         year_profit[year].append((r['exit'] - r['entry']) / r['entry'] * 100)
# for year, profits in year_profit.items():
#     if profits:
#         print(f"{year}: 진입 {len(profits)}회, 총수익률 {sum(profits):.2f}%, 평균수익률 {np.mean(profits):.2f}%")
# ... (이하 전체 실험 print문 주석 처리) ...
# if 'datetime' in df.columns:
#     print(f"\n실험 기간: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
# elif 'timestamp' in df.columns:
#     print(f"\n실험 기간: {pd.to_datetime(df['timestamp'].iloc[0])} ~ {pd.to_datetime(df['timestamp'].iloc[-1])}")
# else:
#     print(f"\n데이터 인덱스(행 번호): {df.index[0]} ~ {df.index[-1]}")
# ... (이하 전체 실험 print문 주석 처리) ...

def run_strategy(df):
    # === 2단계 알고리즘용 함수 정의 (위에서 정의된 함수들 사용) ===
    position = 0
    entry_price = 0
    results = []
    trailing_stop = None
    max_price = 0
    entry_idx = None
    cooldown = 0
    loss_streak = 0
    df['entry_signal'] = False
    for idx in range(30, len(df)-1):
        row = df.iloc[idx]
        if not market_state_filter(row):
            continue
        if not pattern_filter(row, idx, df):
            continue
        if not auxiliary_filter(df, idx):
            continue
        if not timing_filter(df, idx):
            continue
        df.at[idx+1, 'entry_signal'] = True
    for idx in range(31, len(df)):
        row = df.iloc[idx]
        prev_row = df.iloc[idx-1]
        if cooldown > 0:
            cooldown -= 1
        if position == 0 and row['entry_signal'] and cooldown == 0:
            position = 1
            entry_price = row['close'] * (1 + FEE)
            tp, sl = get_tp_sl_v2(prev_row['market_state'])
            trailing_stop = entry_price * (1 - sl)
            max_price = entry_price
            entry_idx = idx
            continue
        if position == 1:
            high = row['high']
            low = row['low']
            if high > entry_price * 1.005:
                trailing_stop = max(trailing_stop, entry_price)
            max_price = max(max_price, high)
            if high >= entry_price * (1 + tp):
                exit_price = entry_price * (1 + tp) * (1 - FEE)
                results.append({'entry': entry_price, 'exit': exit_price, 'type': 'TP', 'entry_idx': entry_idx, 'exit_idx': idx})
                position = 0
                entry_idx = None
                loss_streak = 0
            elif low <= trailing_stop:
                exit_price = trailing_stop * (1 - FEE)
                results.append({'entry': entry_price, 'exit': exit_price, 'type': 'SL', 'entry_idx': entry_idx, 'exit_idx': idx})
                position = 0
                entry_idx = None
                loss_streak += 1
                if loss_streak >= 3:
                    cooldown = 10
            if position == 0 and loss_streak >= 3:
                cooldown = 10
    return results

# 랜덤 구간 실험 함수
def get_random_period_df(df, days):
    df = df.sort_values('datetime').reset_index(drop=True)
    freq_per_day = int(24*60/5)
    period_len = days * freq_per_day
    N = len(df)
    if N < period_len:
        raise ValueError("데이터가 부족합니다.")
    start_idx = random.randint(0, N - period_len)
    end_idx = start_idx + period_len - 1
    return df.iloc[start_idx:end_idx+1].reset_index(drop=True)

# 루프 10회 실험
period_options = list(range(30, 91, 10))  # 30, 40, ..., 90
loop_n = 100  # 실험 루프 수

summary_list = []

for i in range(loop_n):
    days = random.choice(period_options)
    freq_per_day = int(24*60/5)
    period_len = days * freq_per_day
    N = len(df)
    if N < period_len:
        continue
    start_idx = random.randint(0, N - period_len)
    end_idx = start_idx + period_len - 1
    df_sub = df.iloc[start_idx:end_idx+1].reset_index(drop=True)
    if len(df_sub) < period_len:
        continue

    # 전략 실행
    results = run_strategy(df_sub)
    percent_profits = [(r['exit'] - r['entry']) / r['entry'] for r in results]
    cum_returns = [1]
    for p in percent_profits:
        cum_returns.append(cum_returns[-1] * (1 + p))
    cum_returns = np.array(cum_returns)
    cum_return_percent = (cum_returns[-1] - 1) * 100
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    mdd = drawdown.min() * 100
    mean_return = np.mean(percent_profits) if percent_profits else 0
    std_return = np.std(percent_profits) if percent_profits else 0
    win_rate = sum(1 for r in results if r['exit'] > r['entry']) / len(results) * 100 if results else 0
    sharpe = (mean_return / std_return * np.sqrt(len(percent_profits))) if std_return > 0 else 0

    # 시장상태 분포
    state_counts = df_sub['market_state'].value_counts(normalize=True).to_dict()
    bull_ratio = state_counts.get('bull', 0)
    side_ratio = state_counts.get('sideways', 0)
    vol_ratio = state_counts.get('volatile', 0)

    summary_list.append({
        '기간': days,
        '트레이드수': len(results),
        '누적수익률': cum_return_percent,
        'MDD': mdd,
        '승률': win_rate,
        '샤프': sharpe,
        'bull': bull_ratio,
        'sideways': side_ratio,
        'volatile': vol_ratio
    })
    print(f"[{i+1}/{loop_n}] {days}일 | 트레이드수: {len(results)} | 누적수익률: {cum_return_percent:.2f}% | MDD: {mdd:.2f}% | 승률: {win_rate:.2f}% | 샤프: {sharpe:.2f} | bull: {bull_ratio:.2f} | sideways: {side_ratio:.2f} | volatile: {vol_ratio:.2f}")

summary_df = pd.DataFrame(summary_list)
print('\n=== 100회 랜덤 실험 요약 ===')
print(summary_df.describe().loc[['mean', 'std']][['트레이드수','누적수익률','MDD','승률','샤프','bull','sideways','volatile']])

# 시장상태별 성과 차이 요약
for state in ['bull', 'sideways', 'volatile']:
    mask = summary_df[state] > 0.3  # 해당 상태 비율이 30% 이상인 구간만
    print(f'\n[{state}] 비중 30% 이상 구간 평균:')
    print(summary_df[mask][['누적수익률','MDD','승률','샤프']].mean())
