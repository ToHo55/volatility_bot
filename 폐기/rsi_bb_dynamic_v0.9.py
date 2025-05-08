import pandas as pd
import numpy as np
import ta
import random
import itertools
import os
import multiprocessing
import requests
import zipfile
import io
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
import matplotlib.pyplot as plt

FEE = 0.0005  # 업비트 0.05%

# === 누락된 파라미터 정의 ===
RSI_BB_ADX_MIN = 25  # ADX 최소값 (기본값)
RSI_BB_VOL_MULT = 1.5  # 거래량 증가 배수 (기본값)
MEANREV_MAX_HOLD = 20  # Mean Reversion 최대 보유기간 (20봉)

# === 실험용 파라미터 및 옵션 전역 정의 ===
period_options = [5, 10, 15, 20, 30]
loop_n = 500
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

# naive datetime으로 저장된 csv 불러오기
df = pd.read_csv('C:/Cording/volatility_bot/data.csv')
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')  # 자동 변환, 실패시 NaT 처리

# 필요하면 나중에 타임존 부여
# df['datetime'] = df['datetime'].dt.tz_localize('UTC')

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

# === 시장상태 분류: super_low_vol, box, sideways 기준 재조정 ===
def get_market_state(row):
    # 강한 상승장
    if row['close'] > row['ema60'] and row['ema20'] > row['ema60'] and row['adx'] > 20:
        return 'bull'
    # 약한 상승장
    elif row['close'] > row['ema60'] and row['ema20'] > row['ema60']:
        return 'weak_bull'
    # 강한 하락장
    elif row['close'] < row['ema60'] and row['ema20'] < row['ema60']:
        return 'bear'
    # 박스(3,10이동평균 차이 0.5% 이내, 변동성 0.15% 이하)
    elif abs((row['ma3'] - row['ma10']) / row['ma10']) < 0.005 and row['volatility'] < 0.0015:
        return 'box'
    # 횡보장(3,10이동평균 차이 1% 이내, 변동성 0.3% 이하)
    elif abs((row['ma3'] - row['ma10']) / row['ma10']) < 0.01 and row['volatility'] < 0.003:
        return 'sideways'
    # 초저변동성(변동성 0.0007 이하)
    elif row['volatility'] < 0.0007:
        return 'super_low_vol'
    else:
        return 'unknown'

df['market_state'] = df.apply(get_market_state, axis=1)

# === 진입 신호: MACD/RSI/볼밴 상단 중 2개 이상 ===
def entry_signal(idx, df):
    # 진입 신호 발생 후 3봉 내 미진입 시 신호 무효화
    for offset in range(1, 4):
        if idx + offset < len(df):
            if df['close'].iloc[idx + offset] > df['bb_upper'].iloc[idx + offset]:
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
        # ATR 기반 TP/SL, 3봉 내 미도달 시 청산
        tp = row['atr'] * 1.5
        sl = row['atr'] * 1.0
        if row['high'] >= entry_price + tp:
            # 익절
            exit_price = entry_price * (1 + tp) * (1 - FEE)
            results.append({'entry': entry_price, 'exit': exit_price, 'type': 'TP', 'entry_idx': entry_idx, 'exit_idx': idx})
            position = 0
        elif row['low'] <= entry_price - sl:
            # 손절
            exit_price = entry_price * (1 - sl) * (1 - FEE)
            results.append({'entry': entry_price, 'exit': exit_price, 'type': 'SL', 'entry_idx': entry_idx, 'exit_idx': idx})
            position = 0
        elif idx - entry_idx >= 3:
            # 3봉 내 미도달 시 청산
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

# === 전략 파라미터 (Optuna best_params 적용) ===
# Optuna best_params (trial 21 기준)
BULL_MA_DIFF = 0.0020461620165268435
SIDEWAYS_MA_DIFF = 0.006168506375170738
BULL_RSI = 57
SIDEWAYS_RSI = 51
BB_BREAK_BULL = 1.0125486131555348
BB_BREAK_SIDEWAYS = 1.0054017785694391
VOL_MA10_MULT = 2.308202838338314
VOL_MA20_MULT = 1.625284319352012
VOL_ROLLING30_MULT = 1.179154607907264
TP_BULL = 0.02857705999999631
SL_BULL = 0.012006009090580878
TRAILING_BULL = 0.006388067097494876
MEANREV_EMA_DEV = 0.018440473368964094
MEANREV_SL = 0.024970444085685602
MEANREV_TP = 0.012145550183263154
MEANREV_RSI = 48
MEANREV_VOL_MULT = 1.236790158964373
MEANREV_VOLATILITY_MAX = 0.047806268054694324

# === 전략 선택 함수 ===
def select_strategy(row):
    """
    시장 상태에 따라 적절한 전략 선택
    """
    state = row['market_state']
    
    if state == 'bull' and row['adx'] > RSI_BB_ADX_MIN:
        return 'RSI_BB'
    elif state == 'sideways' and row['volatility'] < 0.02:
        return 'MEANREV'
    elif state == 'weak_bull' and row['volatility'] < 0.015:
        return 'MEANREV'
    return None

# === Mean Reversion 전략 개선 ===
def run_meanrev_strategy(df, params=None):
    # params가 없으면 기본값 사용
    if params is None:
        ema_dev = MEANREV_EMA_DEV
        sl = MEANREV_SL
        tp = None  # EMA20 복귀만 사용
        rsi = MEANREV_RSI
        vol_mult = MEANREV_VOL_MULT
        vola_max = MEANREV_VOLATILITY_MAX
    else:
        ema_dev = params.get('meanrev_ema_dev', MEANREV_EMA_DEV)
        sl = params.get('meanrev_sl', MEANREV_SL)
        tp = params.get('meanrev_tp', None)
        rsi = params.get('meanrev_rsi', MEANREV_RSI)
        vol_mult = params.get('meanrev_vol_mult', MEANREV_VOL_MULT)
        vola_max = params.get('meanrev_volatility_max', MEANREV_VOLATILITY_MAX)

    position = 0
    entry_price = 0
    entry_idx = None
    results = []
    for idx in range(20, len(df)):
        row = df.iloc[idx]
        # 진입 조건 체크 (완화)
        if position == 0:
            strategy = select_strategy(row)
            if strategy == 'MEANREV':
                if row['close'] < row['ema20'] * (1 - ema_dev):
                    if row['rsi'] <= rsi + 2:  # rsi 기준 완화
                        if row['volume'] > df['volume'].iloc[idx-5:idx].mean() * (vol_mult * 0.8):  # 거래량 기준 완화
                            if (row['high'] - row['low']) / row['open'] <= vola_max * 1.2:  # 변동성 기준 완화
                                position = 1
                                entry_price = row['close'] * (1 + FEE)
                                entry_idx = idx
                                continue
        # 청산 조건 체크
        if position == 1:
            # TP: EMA20 복귀 또는 entry_price * (1 + tp) 중 먼저 도달
            tp_price = row['ema20']
            if tp is not None:
                tp_price = min(tp_price, entry_price * (1 + tp))
            if row['close'] >= tp_price:
                exit_price = row['close'] * (1 - FEE)
                results.append({
                    'entry': entry_price,
                    'exit': exit_price,
                    'type': 'MEANREV_TP',
                    'entry_idx': entry_idx,
                    'exit_idx': idx
                })
                position = 0
            # SL: 손절
            elif row['low'] <= entry_price * (1 - sl):
                exit_price = entry_price * (1 - sl) * (1 - FEE)
                results.append({
                    'entry': entry_price,
                    'exit': exit_price,
                    'type': 'MEANREV_SL',
                    'entry_idx': entry_idx,
                    'exit_idx': idx
                })
                position = 0
            # TIME: 최대 보유기간
            elif idx - entry_idx >= MEANREV_MAX_HOLD:
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

# === RSI + BB 전략 개선 ===
def improved_entry_signal(idx, df):
    if idx < 2:
        return False
    
    row = df.iloc[idx]
    strategy = select_strategy(row)
    
    if strategy != 'RSI_BB':
        return False
        
    # ADX 필터
    if row['adx'] < RSI_BB_ADX_MIN:
        return False
        
    # MACD 필터
    if row['macd'] < row['macd_signal']:
        return False
    
    # 거래량 필터 강화
    vol_increase = row['volume'] / df['volume'].iloc[idx-5:idx].mean()
    if vol_increase < RSI_BB_VOL_MULT:
        return False
    
    # RSI 상승 추세 확인
    if not (df['rsi'].iloc[idx-2] < df['rsi'].iloc[idx-1] < df['rsi'].iloc[idx]):
        return False
    
    # 볼린저 밴드 상단 돌파 확인
    if row['close'] <= row['bb_upper']:
        return False
    
    return True

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
            if improved_entry_signal(idx, df):
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

# === 기존 전략 실행 함수 ===
def run_strategy(df, params):
    df_copy = df.copy()  # DataFrame 복사
    
    # 기존 전략 실행
    original_results, total, market_fail, pattern_fail, aux_fail, timing_fail = generate_entry_signals(df_copy, params)
    
    # Mean Reversion 전략 실행
    meanrev_results = run_meanrev_strategy(df_copy, params)
    
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

def run_scalping_strategy(df, params=None):
    # TP/SL 파라미터화 (0.7~1%, 0.3~0.5%)
    if params is None:
        scalp_tp = 0.007   # +0.7%
        scalp_sl = 0.003   # -0.3%
    else:
        scalp_tp = params.get('scalp_tp', 0.007)
        scalp_sl = params.get('scalp_sl', 0.003)
        if scalp_tp < scalp_sl:
            scalp_tp = scalp_sl
    position = 0
    entry_price = 0
    entry_idx = None
    results = []
    for idx in range(10, len(df)):
        row = df.iloc[idx]
        # 시장상태 필터: super_low_vol, box, sideways만 진입
        if row['market_state'] not in ['super_low_vol', 'box', 'sideways']:
            continue
        # 진입 조건: rsi 45~65, macd>macd_signal, close>bb_upper 중 2개 이상 만족
        cond1 = 45 < row['rsi'] < 65
        cond2 = row['macd'] > row['macd_signal']
        cond3 = row['close'] > row['bb_upper']
        if position == 0 and sum([cond1, cond2, cond3]) >= 2:
            position = 1
            entry_price = row['close'] * (1 + FEE)
            entry_idx = idx
            continue
        # 청산
        if position == 1:
            # 익절
            if row['high'] >= entry_price * (1 + scalp_tp):
                exit_price = entry_price * (1 + scalp_tp) * (1 - FEE)
                results.append({'entry': entry_price, 'exit': exit_price, 'type': 'SCALP_TP', 'entry_idx': entry_idx, 'exit_idx': idx})
                position = 0
            # 손절
            elif row['low'] <= entry_price * (1 - scalp_sl):
                exit_price = entry_price * (1 - scalp_sl) * (1 - FEE)
                results.append({'entry': entry_price, 'exit': exit_price, 'type': 'SCALP_SL', 'entry_idx': entry_idx, 'exit_idx': idx})
                position = 0
    return results

def run_rsi_bb_experiment():
    print(f'run_rsi_bb_experiment 진입 (pid={os.getpid()})', flush=True)
    summary_rsi_bb = []
    for i in range(loop_n):
        for _ in range(1000):
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
            if bull_ratio >= 0.2:
                results, total, market_fail, pattern_fail, aux_fail, timing_fail = generate_entry_signals(df_sub, params)
                strategy_stats = analyze_strategy_results(results)
                summary = {
                    '기간': days,
                    '전체_트레이드수': len(results),
                    'RSI_BB_트레이드수': strategy_stats['original']['trades'],
                    'RSI_BB_승률': strategy_stats['original']['win_rate'],
                    'RSI_BB_평균수익': strategy_stats['original']['avg_profit'],
                    'bull': bull_ratio,
                }
                summary_rsi_bb.append(summary)
                if (i+1) % 10 == 0:
                    print(f"\n  [RSI+BB] [{i+1}/{loop_n}] {days}일 실험 완료 (bull {bull_ratio:.2f})")
                    print(f"    RSI+BB: {strategy_stats['original']['trades']}건, 승률: {strategy_stats['original']['win_rate']:.1f}%, 평균수익: {strategy_stats['original']['avg_profit']:.2f}%")
                break
    # 총평 출력
    total_trades = sum(s['RSI_BB_트레이드수'] for s in summary_rsi_bb)
    avg_win_rate = np.mean([s['RSI_BB_승률'] for s in summary_rsi_bb if s['RSI_BB_트레이드수'] > 0]) if summary_rsi_bb else 0
    avg_profit = np.mean([s['RSI_BB_평균수익'] for s in summary_rsi_bb if s['RSI_BB_트레이드수'] > 0]) if summary_rsi_bb else 0
    print(f'\n[RSI+BB] 총 실험: {len(summary_rsi_bb)}회, 총 트레이드: {total_trades}건, 평균 승률: {avg_win_rate:.2f}%, 평균 수익률: {avg_profit:.2f}%')
    print(f'[RSI+BB] 실험 완료')

    # 시장상태 분포 시각화
    if 'df_sub' in locals() and 'market_state' in df_sub.columns:
        df_sub['market_state'].value_counts().plot(kind='bar')
        plt.title('시장상태 분포 (실험 구간)')
        plt.show()
    else:
        print("market_state 컬럼이 없거나 df_sub가 정의되지 않았습니다.")

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        # 전체 수익률
        total_return = (results_df['exit'] / results_df['entry']).prod() - 1
        years = 3  # 데이터 기간에 맞게 조정
        annual_return = (1 + total_return) ** (1 / years) - 1
        equity_curve = (results_df['exit'] / results_df['entry']).cumprod()
        peak = equity_curve.cummax()
        mdd = ((equity_curve - peak) / peak).min()
        win_rate = (results_df['exit'] > results_df['entry']).mean()
        returns = results_df['exit'] / results_df['entry'] - 1
        sharpe = returns.mean() / (returns.std() + 1e-8)
        if (annual_return > 0.15 and mdd > -0.1 and win_rate > 0.45 and sharpe > 1.0):
            print("실전 투입 가능!")
        else:
            print("추가 개선 필요")
    else:
        print("트레이드 결과가 없습니다.")

def run_meanrev_experiment():
    print(f'run_meanrev_experiment 진입 (pid={os.getpid()})', flush=True)
    summary_meanrev = []
    for i in range(loop_n):
        for _ in range(1000):
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
            sideways_ratio = state_counts.get('sideways', 0)
            weak_bull_ratio = state_counts.get('weak_bull', 0)
            # === 조건 완화: 0.3 이상으로 변경 ===
            if (sideways_ratio + weak_bull_ratio) >= 0.3:
                meanrev_results = run_meanrev_strategy(df_sub, params)
                trades = len(meanrev_results)
                wins = sum(1 for r in meanrev_results if (r['exit'] - r['entry']) > 0)
                profit = sum((r['exit'] - r['entry']) / r['entry'] for r in meanrev_results)
                win_rate = (wins / trades * 100) if trades > 0 else 0
                avg_profit = (profit / trades * 100) if trades > 0 else 0
                summary = {
                    '기간': days,
                    'MEANREV_트레이드수': trades,
                    'MEANREV_승률': win_rate,
                    'MEANREV_평균수익': avg_profit,
                    'sideways': sideways_ratio,
                    'weak_bull': weak_bull_ratio,
                }
                summary_meanrev.append(summary)
                if (i+1) % 10 == 0:
                    print(f"\n  [MEANREV] [{i+1}/{loop_n}] {days}일 실험 완료 (sideways+weak_bull {sideways_ratio+weak_bull_ratio:.2f})")
                    print(f"    MEANREV: {trades}건, 승률: {win_rate:.1f}%, 평균수익: {avg_profit:.2f}%")
                break
    # 총평 출력
    total_trades = sum(s['MEANREV_트레이드수'] for s in summary_meanrev)
    avg_win_rate = np.mean([s['MEANREV_승률'] for s in summary_meanrev if s['MEANREV_트레이드수'] > 0]) if summary_meanrev else 0
    avg_profit = np.mean([s['MEANREV_평균수익'] for s in summary_meanrev if s['MEANREV_트레이드수'] > 0]) if summary_meanrev else 0
    print(f'\n[MEANREV] 총 실험: {len(summary_meanrev)}회, 총 트레이드: {total_trades}건, 평균 승률: {avg_win_rate:.2f}%, 평균 수익률: {avg_profit:.2f}%')
    print(f'[MEANREV] 실험 완료')

    # 시장상태 분포 시각화
    if 'df_sub' in locals() and 'market_state' in df_sub.columns:
        df_sub['market_state'].value_counts().plot(kind='bar')
        plt.title('시장상태 분포 (실험 구간)')
        plt.show()
    else:
        print("market_state 컬럼이 없거나 df_sub가 정의되지 않았습니다.")

    results_df = pd.DataFrame(results)
    results_df.to_csv('experiment_results.csv', index=False)
    # rolling window 통계
    results_df['rolling_win'] = results_df['profit'].rolling(100).apply(lambda x: (x > 0).mean())

    if not results_df.empty:
        # 전체 수익률
        total_return = (results_df['exit'] / results_df['entry']).prod() - 1
        years = 3  # 데이터 기간에 맞게 조정
        annual_return = (1 + total_return) ** (1 / years) - 1
        equity_curve = (results_df['exit'] / results_df['entry']).cumprod()
        peak = equity_curve.cummax()
        mdd = ((equity_curve - peak) / peak).min()
        win_rate = (results_df['exit'] > results_df['entry']).mean()
        returns = results_df['exit'] / results_df['entry'] - 1
        sharpe = returns.mean() / (returns.std() + 1e-8)
        if (annual_return > 0.15 and mdd > -0.1 and win_rate > 0.45 and sharpe > 1.0):
            print("실전 투입 가능!")
        else:
            print("추가 개선 필요")
    else:
        print("트레이드 결과가 없습니다.")

def run_scalping_experiment():
    print(f'run_scalping_experiment 진입 (pid={os.getpid()})', flush=True)
    import ta
    try:
        # === 1분봉 데이터 불러오기 ===
        df_1min = pd.read_csv('C:/Cording/volatility_bot/data.csv')
        df_1min['datetime'] = pd.to_datetime(df_1min['datetime'])
        # 이하 동일...
        df_1min['rsi'] = ta.momentum.RSIIndicator(df_1min['close'], window=14).rsi()
        df_1min['ma3'] = df_1min['close'].rolling(window=3).mean()
        df_1min['ma10'] = df_1min['close'].rolling(window=10).mean()
        bb = ta.volatility.BollingerBands(df_1min['close'], window=20, window_dev=2)
        df_1min['bb_upper'] = bb.bollinger_hband()
        df_1min['bb_lower'] = bb.bollinger_lband()
        df_1min['vol_ma10'] = df_1min['volume'].rolling(window=10).mean()
        df_1min['vol_ma20'] = df_1min['volume'].rolling(window=20).mean()
        df_1min['volatility'] = df_1min['close'].pct_change().rolling(5).std()
        df_1min['atr'] = ta.volatility.AverageTrueRange(df_1min['high'], df_1min['low'], df_1min['close'], window=10).average_true_range()
        df_1min['ema5'] = df_1min['close'].ewm(span=5).mean()
        df_1min['ema20'] = df_1min['close'].ewm(span=20).mean()
        df_1min['ema60'] = df_1min['close'].ewm(span=60).mean()
        df_1min['macd'] = ta.trend.macd(df_1min['close'])
        df_1min['macd_signal'] = ta.trend.macd_signal(df_1min['close'])
        df_1min['adx'] = ta.trend.ADXIndicator(df_1min['high'], df_1min['low'], df_1min['close'], window=14).adx()
        df_1min = df_1min.dropna().reset_index(drop=True)
        df_1min['market_state'] = df_1min.apply(get_market_state, axis=1)
        # 분포 확인용 print 추가
        print('\n[분포 진단] 1분봉 volatility describe:', flush=True)
        print(df_1min['volatility'].describe(), flush=True)
        print('[분포 진단] volatility<0.1:', (df_1min['volatility'] < 0.1).sum(), '/', len(df_1min), flush=True)
        print('[분포 진단] box abs((ma3-ma10)/ma10)<0.1:', (abs((df_1min['ma3'] - df_1min['ma10']) / df_1min['ma10']) < 0.1).sum(), '/', len(df_1min), flush=True)
        print('[분포 진단] market_state value_counts:', flush=True)
        print(df_1min['market_state'].value_counts(), flush=True)
        print(df_1min['market_state'].value_counts(normalize=True), flush=True)
        # === 1분봉 실험용 period_options 및 freq_per_day ===
        period_options = [60, 120, 180, 240, 300]  # 1~5시간(분 단위)
        freq_per_day = 1  # 1분봉이므로 1
        summary_scalp = []
        found_count = 0
        trade_count = 0
        total_trades = 0
        loop_n = 500
        for i in range(loop_n):
            found = False
            super_low_vol_ratio = 0  # 에러 방지용 기본값
            box_ratio = 0
            for _ in range(1000):
                period_len = random.choice(period_options)
                N = len(df_1min)
                if N < period_len:
                    continue
                start_idx = random.randint(0, N - period_len)
                end_idx = start_idx + period_len - 1
                df_sub = df_1min.iloc[start_idx:end_idx+1].reset_index(drop=True)
                if len(df_sub) < period_len:
                    continue
                state_counts = df_sub['market_state'].value_counts(normalize=True).to_dict()
                super_low_vol_ratio = state_counts.get('super_low_vol', 0)
                box_ratio = state_counts.get('box', 0)
                if (super_low_vol_ratio + box_ratio) >= 0.001:
                    scalp_results = run_scalping_strategy(df_sub, params)
                    trades = len(scalp_results)
                    wins = sum(1 for r in scalp_results if (r['exit'] - r['entry']) > 0)
                    profit = sum((r['exit'] - r['entry']) / r['entry'] for r in scalp_results)
                    win_rate = (wins / trades * 100) if trades > 0 else 0
                    avg_profit = (profit / trades * 100) if trades > 0 else 0
                    summary = {
                        '기간': period_len,
                        'SCALP_트레이드수': trades,
                        'SCALP_승률': win_rate,
                        'SCALP_평균수익': avg_profit,
                        'super_low_vol': super_low_vol_ratio,
                        'box': box_ratio,
                    }
                    summary_scalp.append(summary)
                    found = True
                    found_count += 1
                    if trades > 0:
                        trade_count += 1
                        total_trades += trades
                        log_msg = f"[SCALP] [{i+1}/{loop_n}] 트레이드 {trades}건, 승률: {win_rate:.1f}%, 평균수익: {avg_profit:.2f}% (days={period_len}, super_low_vol={super_low_vol_ratio:.2f}, box={box_ratio:.2f})"
                    else:
                        log_msg = f"[SCALP] [{i+1}/{loop_n}] 조건 충족, 트레이드 없음 (days={period_len}, super_low_vol={super_low_vol_ratio:.2f}, box={box_ratio:.2f})"
                    print(log_msg)
                    break
            if not found:
                log_msg = f"[SCALP] [{i+1}/{loop_n}] 조건에 맞는 구간 없음 (days={period_len}, super_low_vol={super_low_vol_ratio:.2f}, box={box_ratio:.2f})"
                print(log_msg)
        summary_msg = f"[SCALP] 전체 조건 충족: {found_count}회, 트레이드 발생: {trade_count}회, 총 트레이드: {total_trades}건"
        print(summary_msg)
        print(f'\n[SCALP] 실험 완료')

        # === 5분봉 데이터 불러오기 ===
        df_5min = pd.read_pickle('C:/Cording/volatility_bot/0f9039bf4dc80a98c0d73ddae8a32180.pkl')
        if isinstance(df_5min.index, pd.DatetimeIndex):
            df_5min = df_5min.reset_index()
            df_5min = df_5min.rename(columns={'index': 'datetime'})
        df_5min['datetime'] = pd.to_datetime(df_5min['datetime'])
        # 5분봉 지표/상태분류 적용
        df_5min['rsi'] = ta.momentum.RSIIndicator(df_5min['close'], window=14).rsi()
        df_5min['ma3'] = df_5min['close'].rolling(window=3).mean()
        df_5min['ma10'] = df_5min['close'].rolling(window=10).mean()
        bb5 = ta.volatility.BollingerBands(df_5min['close'], window=20, window_dev=2)
        df_5min['bb_upper'] = bb5.bollinger_hband()
        df_5min['bb_lower'] = bb5.bollinger_lband()
        df_5min['volatility'] = df_5min['close'].pct_change().rolling(5).std()
        df_5min = df_5min.dropna().reset_index(drop=True)
        df_5min['market_state'] = df_5min.apply(get_market_state, axis=1)

        # 분석 구간 설정
        start_date = pd.Timestamp('2024-04-28')
        end_date = pd.Timestamp('2025-04-24')

        # 1분봉
        df_1min_period = df_1min[(df_1min['datetime'] >= start_date) & (df_1min['datetime'] < end_date)]
        print('\n[1분봉 시장상태 분포 - 최근 1년]')
        print(df_1min_period['market_state'].value_counts())
        print(df_1min_period['market_state'].value_counts(normalize=True))

        # 5분봉
        df_5min_period = df_5min[(df_5min['datetime'] >= start_date) & (df_5min['datetime'] < end_date)]
        print('\n[5분봉 시장상태 분포 - 최근 1년]')
        print(df_5min_period['market_state'].value_counts())
        print(df_5min_period['market_state'].value_counts(normalize=True))
    except Exception as e:
        print(f'run_scalping_experiment 에러: {e}')

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        # 전체 수익률
        total_return = (results_df['exit'] / results_df['entry']).prod() - 1
        years = 3  # 데이터 기간에 맞게 조정
        annual_return = (1 + total_return) ** (1 / years) - 1
        equity_curve = (results_df['exit'] / results_df['entry']).cumprod()
        peak = equity_curve.cummax()
        mdd = ((equity_curve - peak) / peak).min()
        win_rate = (results_df['exit'] > results_df['entry']).mean()
        returns = results_df['exit'] / results_df['entry'] - 1
        sharpe = returns.mean() / (returns.std() + 1e-8)
        if (annual_return > 0.15 and mdd > -0.1 and win_rate > 0.45 and sharpe > 1.0):
            print("실전 투입 가능!")
        else:
            print("추가 개선 필요")
    else:
        print("트레이드 결과가 없습니다.")

def strategy_selector(row):
    if row['market_state'] == 'bull':
        return 'RSI_BB'
    elif row['market_state'] in ['sideways', 'box']:
        return 'MEANREV'
    else:
        return None

# 메인 엔진 루프 예시
for idx, row in df.iterrows():
    strategy = strategy_selector(row)
    if strategy == 'RSI_BB':
        # RSI+BB 전략 실행
        pass
    elif strategy == 'MEANREV':
        # MeanReversion 전략 실행
        pass

def load_data(symbol, tf):
    file_path = f'C:/Cording/volatility_bot/data_{symbol}_{tf}.csv'
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"{file_path} 로딩 실패: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    print(f"\n{'='*20} 3년치 통합 데이터 실험 시작 (시장상태별 전략 분리) {'='*20}")
    import multiprocessing
    p1 = multiprocessing.Process(target=run_rsi_bb_experiment)
    p2 = multiprocessing.Process(target=run_meanrev_experiment)
    p3 = multiprocessing.Process(target=run_scalping_experiment)
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    print('\n=== 모든 전략 병렬 실험 완료 ===')
    print(df['market_state'].value_counts(normalize=True))

for symbol in ['BTC', 'ETH', 'XRP']:
    for tf in ['5min', '15min', '1h']:
        df = load_data(symbol, tf)
        # 동일 전략 반복 적용 및 결과 저장
