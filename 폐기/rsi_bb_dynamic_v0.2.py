import pandas as pd
import numpy as np
import ta
import random
import itertools
import os

FEE = 0.0005  # 업비트 0.05%

def analyze_strategy_results(results):
    """
    전략별 거래 결과를 분석하는 함수
    """
    # 빈 결과 처리
    if not results:
        return {
            'original': {'trades': 0, 'wins': 0, 'profit': 0, 'win_rate': 0, 'avg_profit': 0},
            'meanrev': {'trades': 0, 'wins': 0, 'profit': 0, 'win_rate': 0, 'avg_profit': 0}
        }
    
    # 전략별 분석
    strategy_stats = {
        'original': {'trades': 0, 'wins': 0, 'profit': 0},
        'meanrev': {'trades': 0, 'wins': 0, 'profit': 0}
    }
    
    # 각 거래 결과 분석
    for trade in results:
        # 전략 구분
        is_meanrev = 'MEANREV' in trade['type']
        strategy = 'meanrev' if is_meanrev else 'original'
        
        # 수익률 계산
        profit = (trade['exit'] - trade['entry']) / trade['entry']
        
        # 통계 업데이트
        stats = strategy_stats[strategy]
        stats['trades'] += 1
        stats['wins'] += 1 if profit > 0 else 0
        stats['profit'] += profit
    
    # 최종 통계 계산
    for strategy, stats in strategy_stats.items():
        if stats['trades'] > 0:
            stats['win_rate'] = stats['wins'] / stats['trades'] * 100
            stats['avg_profit'] = stats['profit'] / stats['trades'] * 100
        else:
            stats['win_rate'] = 0
            stats['avg_profit'] = 0
    
    return strategy_stats

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
# === ATR(5) 지표 추가 ===
df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=5).average_true_range()
df['ema5'] = df['close'].ewm(span=5).mean()
df['ema20'] = df['close'].ewm(span=20).mean()
df['ema60'] = df['close'].ewm(span=60).mean()
# === MACD, ADX 지표 추가 ===
df['macd'] = ta.trend.macd(df['close'])
df['macd_signal'] = ta.trend.macd_signal(df['close'])
df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()

# NaN 제거 (여기서!)
df = df.dropna().reset_index(drop=True)

# === 시장상태 분류: 장기 MA 기반 고도화 ===
def get_market_state(row):
    # 장기 MA 위에 있고, ema20 > ema60이면 bull
    if row['close'] > row['ema60'] and row['ema20'] > row['ema60']:
        return 'bull'
    elif abs((row['ma3'] - row['ma10']) / row['ma10']) < 0.008:
        return 'sideways'
    elif row['close'] < row['ema60'] and row['ema20'] < row['ema60']:
        return 'bear'
    else:
        return 'unknown'

df['market_state'] = df.apply(get_market_state, axis=1)

# === 진입 신호: MACD/RSI/볼밴 상단 중 2개 이상 ===
def entry_signal(idx, df):
    if idx < 1:
        return False
    row = df.iloc[idx]
    prev = df.iloc[idx-1]
    cond1 = prev['macd'] <= prev['macd_signal'] and row['macd'] > row['macd_signal']
    cond2 = row['rsi'] > 52
    cond3 = row['close'] > row['bb_upper']
    if sum([cond1, cond2, cond3]) >= 2:
        return True
    return False

# === 메인 엔진 루프 (진입 신호/청산) ===
df['entry_signal'] = False
for idx in range(1, len(df)):
    if entry_signal(idx, df):
        df.at[idx, 'entry_signal'] = True

position = 0
entry_price = 0
max_price = 0
entry_idx = None
results = []
for idx in range(1, len(df)):
    row = df.iloc[idx]
    # 시장상태별 TP/SL 구조 개선
    if row['market_state'] == 'bull':
        tp, sl = 0.02, 0.01  # 2:1 비율
    elif row['market_state'] == 'sideways':
        tp, sl = 0.01, 0.008
    else:
        tp, sl = 0.01, 0.01
    if position == 0 and row['entry_signal']:
        position = 1
        entry_price = row['close'] * (1 + FEE)
        max_price = entry_price
        entry_idx = idx
        continue
    if position == 1:
        max_price = max(max_price, row['high'])
        # 익절
        if row['high'] >= entry_price * (1 + tp):
            exit_price = entry_price * (1 + tp) * (1 - FEE)
            results.append({'entry': entry_price, 'exit': exit_price, 'type': 'TP', 'entry_idx': entry_idx, 'exit_idx': idx})
            position = 0
        # 손절
        elif row['low'] <= entry_price * (1 - sl):
            exit_price = entry_price * (1 - sl) * (1 - FEE)
            results.append({'entry': entry_price, 'exit': exit_price, 'type': 'SL', 'entry_idx': entry_idx, 'exit_idx': idx})
            position = 0
        # 최대 보유기간 10봉 초과시 breakeven 청산
        elif idx - entry_idx >= 10:
            exit_price = row['close'] * (1 - FEE)
            results.append({'entry': entry_price, 'exit': exit_price, 'type': 'TIME', 'entry_idx': entry_idx, 'exit_idx': idx})
            position = 0

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

def market_state_filter(row, params):
    if row['market_state'] not in ['bull', 'sideways', 'volatile']:
        return False
    if row['market_state'] == 'bull':
        ma_short = row['ma3']
        ma_long = row['ma10']
        ma_diff = (ma_short - ma_long) / ma_long if ma_long != 0 else 0
        if ma_diff < params['bull_ma_diff']:
            return False
    if row['market_state'] == 'sideways':
        # 예: 볼밴 하단 3% 이탈 + RSI < 35 + 거래량 2배 이상 등
        if not (row['close'] < row['bb_lower'] * 0.97 and row['rsi'] < 35 and row['volume'] > df['volume'].iloc[idx-5:idx].mean() * 2):
            return False
    if row['market_state'] == 'volatile':
        cond1 = (row['high'] - row['low']) / row['open'] > 0.02
        if not cond1:
            return False
    return True

def get_tp_sl_v4(market_state, params):
    if market_state == 'bull':
        return params['tp_bull'], params['sl_bull'], params['trailing_bull']
    elif market_state == 'sideways':
        return 0.01, 0.003, 0.002
    elif market_state == 'volatile':
        return 0.02, 0.007, 0.003
    else:
        return 0.01, 0.01, 0.003

def pattern_filter(row, idx, df, params):
    close = row['close']
    upper = row['bb_upper']
    lower = row['bb_lower']
    rsi = row['rsi']
    state = row['market_state']
    if state == 'bull':
        cond1 = (close > upper * params['bb_break_bull'] and rsi > params['bull_rsi'])
        cond2 = (idx >= 2 and df['rsi'].iloc[idx-2] < df['rsi'].iloc[idx-1] < df['rsi'].iloc[idx])
        cond3 = (close > df['high'].iloc[max(0, idx-10):idx].max())
        if not (cond1 or cond2 or cond3):
            return False
    elif state == 'sideways':
        if not (close < lower * params['bb_break_sideways'] and rsi < params['sideways_rsi']):
            return False
    elif state == 'volatile':
        cond1 = (close < lower * 1.01 and rsi < 58)
        cond2 = (row['high'] - row['low']) / row['open'] > 0.02
        if not (cond1 or cond2):
            return False
    if idx < 2:
        return False
    count = 0
    for i in range(3):
        if df['close'].iloc[idx-i] > df['open'].iloc[idx-i]:
            count += 1
    if count < 1:
        return False
    return True

def auxiliary_filter(df, idx, params):
    if idx < 30:
        return False
    v = df['volume'].iloc[idx]
    if not (v > df['vol_ma10'].iloc[idx]*params['vol_ma10_mult'] and v > df['vol_ma20'].iloc[idx]*params['vol_ma20_mult'] and v > df['volume'].rolling(30).mean().iloc[idx]*params['vol_rolling30_mult']):
        return False
    if not (df['rsi'].iloc[idx-1] < df['rsi'].iloc[idx]):
        return False
    return True

def timing_filter(df, idx):
    # 4차: 신호 다음 봉의 open > 신호봉 close
    if idx+1 >= len(df):
        return False
    return df['open'].iloc[idx+1] > df['close'].iloc[idx]

def strong_rsi_trend(df, idx):
    if idx < 4:
        return False
    count = 0
    for i in range(5):
        if df['rsi'].iloc[idx-i] > df['rsi'].iloc[idx-i-1]:
            count += 1
    return count >= 3

def generate_entry_signals(df, params):
    total, market_fail, pattern_fail, aux_fail, timing_fail = 0, 0, 0, 0, 0
    df['entry_signal'] = False
    results = []
    position = 0
    entry_price = 0
    max_price = 0
    entry_idx = None
    
    for idx in range(30, len(df)-1):
        row = df.iloc[idx]
        total += 1
        
        # 진입 신호 생성
        if position == 0:
            if not market_state_filter(row, params):
                market_fail += 1
                continue
            if not pattern_filter(row, idx, df, params):
                pattern_fail += 1
                continue
            if not auxiliary_filter(df, idx, params):
                aux_fail += 1
                continue
            if not timing_filter(df, idx):
                timing_fail += 1
                continue
            if strong_rsi_trend(df, idx):
                position = 1
                entry_price = row['close'] * (1 + FEE)
                max_price = entry_price
                entry_idx = idx
                continue
        
        # 포지션 관리
        if position == 1:
            tp, sl, trailing = get_tp_sl_v4(row['market_state'], params)
            
            # 익절
            if row['high'] >= entry_price * (1 + tp):
                exit_price = entry_price * (1 + tp) * (1 - FEE)
                results.append({
                    'entry': entry_price,
                    'exit': exit_price,
                    'type': 'TP',
                    'entry_idx': entry_idx,
                    'exit_idx': idx
                })
                position = 0
            # 손절
            elif row['low'] <= entry_price * (1 - sl):
                exit_price = entry_price * (1 - sl) * (1 - FEE)
                results.append({
                    'entry': entry_price,
                    'exit': exit_price,
                    'type': 'SL',
                    'entry_idx': entry_idx,
                    'exit_idx': idx
                })
                position = 0
            # 트레일링 스탑
            elif max_price >= entry_price * 1.015 and row['low'] <= max_price * (1 - trailing):
                exit_price = max_price * (1 - trailing) * (1 - FEE)
                results.append({
                    'entry': entry_price,
                    'exit': exit_price,
                    'type': 'TRAIL',
                    'entry_idx': entry_idx,
                    'exit_idx': idx
                })
                position = 0
            
            if position == 1:
                max_price = max(max_price, row['high'])
    
    return results, total, market_fail, pattern_fail, aux_fail, timing_fail

# === Mean Reversion 전략 파라미터 ===
MEANREV_EMA_DEV = 0.02  # EMA 이탈 기준 (2%)
MEANREV_SL = 0.03      # 손절 기준 (3%)
MEANREV_MAX_HOLD = 20  # 최대 보유기간 (20봉)

# === Mean Reversion 전략 실행 함수 ===
def run_meanrev_strategy(df):
    """
    Mean Reversion 전략을 실행하는 함수
    - 20EMA 대비 2% 이상 하락 시 매수
    - EMA 복귀 시 매도
    - 3% 손절
    - 최대 20봉 보유
    """
    position = 0
    entry_price = 0
    entry_idx = None
    results = []
    
    for idx in range(20, len(df)):
        row = df.iloc[idx]
        
        # 진입 조건 체크
        if position == 0:
            # 기본 필터: 20EMA -2% 이탈
            if row['close'] < row['ema20'] * 0.98:
                # 추가 필터1: RSI 40 이하
                if row['rsi'] <= 40:
                    # 추가 필터2: 거래량 증가
                    if row['volume'] > df['volume'].iloc[idx-5:idx].mean() * 1.5:
                        # 추가 필터3: 변동성 제한
                        if (row['high'] - row['low']) / row['open'] <= 0.03:
                            position = 1
                            entry_price = row['close'] * (1 + FEE)
                            entry_idx = idx
                            continue
        
        # 청산 조건 체크
        if position == 1:
            # 청산 조건1: EMA20 복귀
            if row['close'] >= row['ema20']:
                exit_price = row['close'] * (1 - FEE)
                results.append({
                    'entry': entry_price,
                    'exit': exit_price,
                    'type': 'MEANREV_TP',
                    'entry_idx': entry_idx,
                    'exit_idx': idx
                })
                position = 0
            # 청산 조건2: 손절 (-3%)
            elif row['low'] <= entry_price * 0.97:
                exit_price = entry_price * 0.97 * (1 - FEE)
                results.append({
                    'entry': entry_price,
                    'exit': exit_price,
                    'type': 'MEANREV_SL',
                    'entry_idx': entry_idx,
                    'exit_idx': idx
                })
                position = 0
            # 청산 조건3: 최대 보유기간 (20봉)
            elif idx - entry_idx >= 20:
                exit_price = row['close'] * (1 - FEE)
                results.append({
                    'entry': entry_price,
                    'exit': exit_price,
                    'type': 'MEANREV_TIME',
                    'entry_idx': entry_idx,
                    'exit_idx': idx
                })
                position = 0
    
    return results

# === 기존 전략 실행 함수 ===
def run_strategy(df, params):
    df_copy = df.copy()  # DataFrame 복사
    
    # 기존 전략 실행
    original_results, total, market_fail, pattern_fail, aux_fail, timing_fail = generate_entry_signals(df_copy, params)
    
    # Mean Reversion 전략 실행
    meanrev_results = run_meanrev_strategy(df_copy)
    
    # 결과 통합 (시간순 정렬)
    all_results = original_results + meanrev_results if meanrev_results else original_results
    if all_results:
        all_results.sort(key=lambda x: x['entry_idx'])
    
    return all_results, total, market_fail, pattern_fail, aux_fail, timing_fail

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

# === 3년치 통합 데이터 실험 ===
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', '0f9039bf4dc80a98c0d73ddae8a32180.pkl')
df = pd.read_pickle(file_path)
if not isinstance(df, pd.DataFrame):
    print(f"{file_path} 파일은 DataFrame이 아닙니다. (type: {type(df)})")
    exit()
if 'datetime' not in df.columns:
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'datetime'})
    else:
        print(f"{file_path} 파일에 datetime 정보가 없습니다.")
        exit()
df['datetime'] = pd.to_datetime(df['datetime'])
# 이하 기존 지표 계산, 실험 루프, 결과 출력 동일하게 적용
# ... 기존 for coin in coin_list: 루프 및 내부 내용 제거 ...
period_options = [10, 15, 20, 30, 40]
loop_n = 100
bull_threshold = 0.3
param_grid = {
    'bull_ma_diff': [0.002, 0.0025, 0.003, 0.0035, 0.004],
    'sideways_ma_diff': [0.005, 0.006, 0.007, 0.008],
    'bull_rsi': [55, 58, 60, 62, 65],
    'sideways_rsi': [48, 50, 52],
    'bb_break_bull': [1.003, 1.004, 1.005, 1.015],
    'bb_break_sideways': [1.005, 1.008, 1.01],
    'vol_ma10_mult': [2, 2.2, 2.5, 3],
    'vol_ma20_mult': [1.5, 1.8, 2],
    'vol_rolling30_mult': [1.1, 1.2, 1.3],
    'tp_bull': [0.02, 0.025, 0.03],
    'sl_bull': [0.01, 0.012],
    'trailing_bull': [0.005, 0.007],
}
params = {k: v[0] for k, v in param_grid.items()}
# 지표 계산
print(f"\n{'='*20} 3년치 통합 데이터 실험 시작 {'='*20}")
df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
df['ma3'] = df['close'].rolling(window=3).mean()
df['ma10'] = df['close'].rolling(window=10).mean()
bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
df['bb_upper'] = bb.bollinger_hband()
df['bb_lower'] = bb.bollinger_lband()
df['vol_ma10'] = df['volume'].rolling(window=10).mean()
df['vol_ma20'] = df['volume'].rolling(window=20).mean()
df['volatility'] = df['close'].pct_change().rolling(5).std()
df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=5).average_true_range()
df['ema5'] = df['close'].ewm(span=5).mean()
df['ema20'] = df['close'].ewm(span=20).mean()
df['ema60'] = df['close'].ewm(span=60).mean()
df['macd'] = ta.trend.macd(df['close'])
df['macd_signal'] = ta.trend.macd_signal(df['close'])
df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
df = df.dropna().reset_index(drop=True)
df['market_state'] = df.apply(get_market_state, axis=1)

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
    state_counts = df_sub['market_state'].value_counts(normalize=True).to_dict()
    bull_ratio = state_counts.get('bull', 0)
    if bull_ratio < bull_threshold:
        continue
    results, total, market_fail, pattern_fail, aux_fail, timing_fail = run_strategy(df_sub, params)
    strategy_stats = analyze_strategy_results(results)
    
    # 전략별 성과 기록
    summary = {
        '기간': days,
        '전체_트레이드수': len(results),
        'RSI_BB_트레이드수': strategy_stats['original']['trades'],
        'MEANREV_트레이드수': strategy_stats['meanrev']['trades'],
        'RSI_BB_승률': strategy_stats['original']['win_rate'],
        'MEANREV_승률': strategy_stats['meanrev']['win_rate'],
        'RSI_BB_평균수익': strategy_stats['original']['avg_profit'],
        'MEANREV_평균수익': strategy_stats['meanrev']['avg_profit'],
        'bull': bull_ratio,
        'sideways': state_counts.get('sideways', 0),
        'volatile': state_counts.get('volatile', 0)
    }
    summary_list.append(summary)
    
    if (i+1) % 10 == 0:
        print(f"\n  [{i+1}/{loop_n}] {days}일 실험 완료")
        print(f"    RSI+BB: {strategy_stats['original']['trades']}건, "
              f"승률: {strategy_stats['original']['win_rate']:.1f}%, "
              f"평균수익: {strategy_stats['original']['avg_profit']:.2f}%")
        print(f"    MEANREV: {strategy_stats['meanrev']['trades']}건, "
              f"승률: {strategy_stats['meanrev']['win_rate']:.1f}%, "
              f"평균수익: {strategy_stats['meanrev']['avg_profit']:.2f}%")

# === 결과 출력 ===
summary_df = pd.DataFrame(summary_list)
print(f'\n=== 전략별 성과 비교 (평균) ===')
if not summary_df.empty:
    print('\nRSI + BB 전략:')
    print(f"트레이드수: {summary_df['RSI_BB_트레이드수'].mean():.1f}")
    print(f"승률: {summary_df['RSI_BB_승률'].mean():.1f}%")
    print(f"평균수익: {summary_df['RSI_BB_평균수익'].mean():.2f}%")
    
    print('\nMean Reversion 전략:')
    print(f"트레이드수: {summary_df['MEANREV_트레이드수'].mean():.1f}")
    print(f"승률: {summary_df['MEANREV_승률'].mean():.1f}%")
    print(f"평균수익: {summary_df['MEANREV_평균수익'].mean():.2f}%")
else:
    print('실험 결과가 없습니다.')

# === Optuna 자동 파라미터 최적화 코드 추가 ===
try:
    import optuna
except ImportError:
    print('optuna가 설치되어 있지 않습니다. 설치하려면: pip install optuna')
    optuna = None

if optuna:
    def objective(trial):
        # 주요 파라미터 범위 정의 (필요시 추가/수정)
        params = {
            'bull_ma_diff': trial.suggest_float('bull_ma_diff', 0.002, 0.004),
            'sideways_ma_diff': trial.suggest_float('sideways_ma_diff', 0.005, 0.008),
            'bull_rsi': trial.suggest_int('bull_rsi', 55, 65),
            'sideways_rsi': trial.suggest_int('sideways_rsi', 48, 52),
            'bb_break_bull': trial.suggest_float('bb_break_bull', 1.003, 1.015),
            'bb_break_sideways': trial.suggest_float('bb_break_sideways', 1.005, 1.01),
            'vol_ma10_mult': trial.suggest_float('vol_ma10_mult', 2, 3),
            'vol_ma20_mult': trial.suggest_float('vol_ma20_mult', 1.5, 2),
            'vol_rolling30_mult': trial.suggest_float('vol_rolling30_mult', 1.1, 1.3),
            'tp_bull': trial.suggest_float('tp_bull', 0.02, 0.04),
            'sl_bull': trial.suggest_float('sl_bull', 0.008, 0.015),
            'trailing_bull': trial.suggest_float('trailing_bull', 0.004, 0.01),
        }
        # 랜덤 구간 여러 번 반복 후 평균 누적수익률 반환
        period_options = [30, 40, 50, 60, 70, 80, 90]
        loop_n = 100  # optuna trial당 반복 횟수 (속도 고려, 필요시 증가)
        total_return = 0
        valid_trials = 0
        for _ in range(loop_n):
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
            results = run_strategy(df_sub, params)
            if results:
                if isinstance(results[0], list):
                    results = [item for sublist in results if isinstance(sublist, list) for item in sublist]
                elif not isinstance(results[0], dict):
                    results = []
            percent_profits = [(r['exit'] - r['entry']) / r['entry'] for r in results]
            cum_returns = [1]
            for p in percent_profits:
                cum_returns.append(cum_returns[-1] * (1 + p))
            cum_returns = np.array(cum_returns)
            cum_return_percent = (cum_returns[-1] - 1) * 100
            total_return += cum_return_percent
            valid_trials += 1
        if valid_trials == 0:
            return -9999  # 데이터 부족시 큰 음수 반환
        return total_return / valid_trials  # 평균 누적수익률

    print('\n=== Optuna 자동 파라미터 최적화 시작 ===')
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)  # n_trials는 필요시 조정
    print('\n=== Optuna 최적 파라미터 ===')
    print(study.best_params)
    print(f'최고 평균 누적수익률: {study.best_value:.2f}%')

# === 추세+조정+반등 전략 진입 신호 ===
def trend_entry_signal(idx, df):
    if idx < 10:
        return False
    row = df.iloc[idx]
    # 시장 필터: Uptrend
    if not (row['ema20'] > row['ema60'] and row['ema20'] > df['ema20'].iloc[idx-1] and row['ema5'] > row['ema20']):
        return False
    # 단기 조정: 2봉 이상 음봉 + 전체 하락폭 1.5% 이내
    down_count = 0
    for i in range(1, 4):
        if df['close'].iloc[idx-i] < df['open'].iloc[idx-i]:
            down_count += 1
    if down_count < 2:
        return False
    corr_start = idx - down_count
    corr_end = idx - 1
    corr_drop = (df['close'].iloc[corr_end] - df['close'].iloc[corr_start]) / df['close'].iloc[corr_start]
    if corr_drop < -0.015:
        return False
    # 매수 트리거: 당일 종가가 직전 고가 대비 +0.5% 이상 상승
    if row['close'] < df['high'].iloc[idx-1] * 1.005:
        return False
    # 거래량 필터: 최근 5봉 평균 대비 1.2배 이상
    if row['volume'] < df['volume'].iloc[idx-5:idx].mean() * 1.2:
        return False
    # RSI 필터
    if row['rsi'] <= 55:
        return False
    # 예외 필터1: 최근 10봉 최고가 대비 -5% 이상 급락 시 진입 금지
    max10 = df['high'].iloc[idx-10:idx].max()
    if row['close'] < max10 * 0.95:
        return False
    # 예외 필터2: 일일 변동폭 3% 넘으면 진입 금지
    if (row['high'] - row['low']) / row['open'] > 0.03:
        return False
    return True

# === TP/SL/트레일링 조건 ===
def get_tp_sl_trailing(row, is_volatile):
    if is_volatile:
        return 0.015, 0.01, 0.007  # TP, SL, trailing
    else:
        return 0.025, 0.012, 0.007

# === 메인 엔진 루프 (추세+조정+반등 전략) ===
df['entry_signal'] = False
for idx in range(10, len(df)):
    if trend_entry_signal(idx, df):
        df.at[idx, 'entry_signal'] = True

position = 0
entry_price = 0
max_price = 0
entry_idx = None
results = []
for idx in range(10, len(df)):
    row = df.iloc[idx]
    is_volatile = (row['high'] - row['low']) / row['open'] > 0.02
    tp, sl, trailing = get_tp_sl_trailing(row, is_volatile)
    if position == 0 and row['entry_signal']:
        position = 1
        entry_price = row['close'] * (1 + FEE)
        max_price = entry_price
        entry_idx = idx
        continue
    if position == 1:
        max_price = max(max_price, row['high'])
        # 익절
        if row['high'] >= entry_price * (1 + tp):
            exit_price = entry_price * (1 + tp) * (1 - FEE)
            results.append({'entry': entry_price, 'exit': exit_price, 'type': 'TP', 'entry_idx': entry_idx, 'exit_idx': idx})
            position = 0
        # 트레일링 스탑
        elif max_price >= entry_price * 1.015 and row['low'] <= max_price * (1 - trailing):
            exit_price = max_price * (1 - trailing) * (1 - FEE)
            results.append({'entry': entry_price, 'exit': exit_price, 'type': 'TRAIL', 'entry_idx': entry_idx, 'exit_idx': idx})
            position = 0
        # 손절
        elif row['low'] <= entry_price * (1 - sl):
            exit_price = entry_price * (1 - sl) * (1 - FEE)
            results.append({'entry': entry_price, 'exit': exit_price, 'type': 'SL', 'entry_idx': entry_idx, 'exit_idx': idx})
            position = 0
