#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import logging
import pyupbit
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 환경 설정
load_dotenv()
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

KST = ZoneInfo("Asia/Seoul")

# 로깅 설정
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/spike_bot.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

class BaseStrategy(ABC):
    def __init__(self):
        self.fees = {
            'buy_fee': 0.0005,    # 매수 수수료 0.05%
            'sell_fee': 0.0005,   # 매도 수수료 0.05%
            'slippage': 0.001     # 슬리피지 0.1%
        }
        self.trend_ma = 20        # 추세 판단용 이동평균
        self.pattern_window = 20  # 패턴 인식용 기간
        self.trend_window = 20  # 추세 판단을 위한 기간
        self.ma_window = 20     # 이동평균선 기간
        self.rsi_window = 14    # RSI 기간

    def calculate_profit_ratio(self, current_price: float, entry_price: float) -> float:
        """수수료와 슬리피지를 포함한 실제 수익률 계산"""
        buy_price = entry_price * (1 + self.fees['buy_fee'] + self.fees['slippage'])
        sell_price = current_price * (1 - self.fees['sell_fee'] - self.fees['slippage'])
        return (sell_price / buy_price) - 1

    def is_uptrend(self, df: pd.DataFrame) -> bool:
        """상승 추세 여부 확인"""
        ma = df['close'].rolling(window=self.trend_ma).mean()
        ma_slope = (ma.iloc[-1] - ma.iloc[-2]) / ma.iloc[-2]
        return ma_slope > 0

    def find_recent_high(self, df: pd.DataFrame, lookback: int = 20) -> float:
        """최근 고점 찾기"""
        return df['high'].rolling(window=lookback).max().iloc[-1]

    def is_breakout(self, df: pd.DataFrame) -> bool:
        """이전 고점 돌파 여부"""
        recent_high = self.find_recent_high(df, self.pattern_window)
        return df['close'].iloc[-1] > recent_high

    @abstractmethod
    def evaluate_entry(self, df: pd.DataFrame) -> bool:
        pass

    @abstractmethod
    def evaluate_exit(self, df: pd.DataFrame, entry_price: float, hold_time: float) -> Tuple[bool, str]:
        pass

class PositionManager:
    def __init__(self):
        self.current_position = None
        self.strategy_priorities = {
            'VolatilityMAStrategy': 1,       # 최우선 순위
            'SpikeStrategy': 2,              # 2순위
            'VolatilityMALiteStrategy': 3,   # 3순위
            'RSIReversalStrategy': 4         # 4순위
        }
    
    def can_enter(self, strategy_name: str) -> bool:
        if self.current_position is None:
            return True
        return (self.strategy_priorities[strategy_name] < 
                self.strategy_priorities[self.current_position])
    
    def enter_position(self, strategy_name: str):
        self.current_position = strategy_name
    
    def exit_position(self):
        self.current_position = None
    
    def get_position_size(self, strategy_name: str, base_size: float, weights: Dict[str, float]) -> float:
        if strategy_name not in weights:
            return base_size
        return base_size * weights[strategy_name]

class VolatilityMAStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.k = 0.45
        self.ma_period = 20
        self.take_profit = 0.025
        self.stop_loss = -0.01
        self.max_hold_time = 240
        self.min_hold_time = 10
        self.volume_ma = 20
        self.min_volume_ratio = 1.1
        self.trend_window = 5
        self.adx_period = 14
        self.adx_threshold = 20
        self.bb_multiplier = 1.8
        self.rsi_upper = 75
        self.rsi_lower = 25
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.dynamic_tp_multiplier = 3.0
        self.atr_period = 20
        self.extended_hold_multiplier = 2.5
        self.min_trend_strength = 0.0003
        self.partial_tp_ratio = 0.6
        self.trailing_stop = 0.004
        self.min_extra_conditions = 2
        self.min_entry_interval = 1800

    def calculate_macd(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD 계산"""
        exp1 = df['close'].ewm(span=self.macd_fast).mean()
        exp2 = df['close'].ewm(span=self.macd_slow).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.macd_signal).mean()
        hist = macd - signal
        return macd, signal, hist

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """ATR 계산"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        return atr

    def evaluate_entry(self, df: pd.DataFrame) -> bool:
        if df is None or len(df) < max(self.ma_period, self.macd_slow) + 1:
            return False
            
        df = df.copy()
        
        # 기본 지표 계산
        df['high_low_range'] = df['high'] - df['low']
        df['target'] = df['open'] + df['high_low_range'].shift(1) * self.k
        df['ma'] = df['close'].rolling(window=self.ma_period).mean()
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        macd, signal, hist = self.calculate_macd(df)
        
        # 추세 강도 계산
        ma_slope = (df['ma'].iloc[-1] - df['ma'].iloc[-2]) / df['ma'].iloc[-2]
        
        # ADX 계산
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=self.adx_period).mean()
        
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)
        
        plus_di = 100 * (plus_dm.rolling(window=self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=self.adx_period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=self.adx_period).mean()
        
        # 볼린저 밴드
        df['bb_ma'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_ma'] + self.bb_multiplier * df['bb_std']
        df['bb_lower'] = df['bb_ma'] - self.bb_multiplier * df['bb_std']
        
        # 추가 지표: 가격 모멘텀
        df['momentum'] = df['close'] / df['close'].shift(5) - 1
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # MACD 골든크로스/데드크로스 확인
        macd_cross_up = (macd.iloc[-1] > signal.iloc[-1]) and (macd.iloc[-2] <= signal.iloc[-2])
        
        # 기본 조건 (RSI 제외)
        base_conditions = [
            last['close'] > last['target'],           # 변동성 돌파
            last['close'] > last['ma'],              # 이동평균선 위
            last['volume_ratio'] > self.min_volume_ratio,  # 거래량 증가
            ma_slope > self.min_trend_strength       # 최소 추세 강도
        ]
        
        # 보조 조건 (2개 이상 만족)
        extra_conditions = [
            self.is_breakout(df),                     # 이전 고점 돌파
            last['close'] > last['bb_upper'],         # 볼린저 상단 돌파
            (last['close'] > last['bb_ma'] and        # 중심선 위 + 거래량 급증
             last['volume_ratio'] > 1.8),             # 거래량 기준 완화 (2.0 → 1.8)
            macd_cross_up,                            # MACD 골든크로스
            adx.iloc[-1] > self.adx_threshold,        # ADX 기준
            hist.iloc[-1] > 0,                        # MACD 히스토그램 양수
            last['rsi'] < self.rsi_upper,             # RSI 상단 미만
            last['rsi'] > self.rsi_lower              # RSI 하단 초과
        ]
        
        return all(base_conditions) and sum(extra_conditions) >= self.min_extra_conditions

    def evaluate_exit(self, df: pd.DataFrame, entry_price: float, hold_time: float) -> Tuple[bool, str]:
        if df is None or df.empty or hold_time < self.min_hold_time:
            return False, ""
            
        current_price = df.iloc[-1]['close']
        profit_ratio = self.calculate_profit_ratio(current_price, entry_price)
        
        # ATR 기반 동적 익절 개선
        atr = self.calculate_atr(df)
        current_atr = atr.iloc[-1]
        atr_based_tp = (current_atr / entry_price) * self.dynamic_tp_multiplier
        
        # 추세 강도에 따른 동적 TP 조정
        ma_slope = (df['close'].rolling(window=self.trend_window).mean().diff() / 
                   df['close'].rolling(window=self.trend_window).mean()).iloc[-1]
        trend_adjusted_tp = atr_based_tp * (1 + abs(ma_slope) * 10)  # 추세가 강할수록 TP 증가
        
        # 개선된 익절 로직
        if profit_ratio >= trend_adjusted_tp:
            return True, "추세 기반 동적 익절"
            
        # RSI 계산
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        macd, signal, hist = self.calculate_macd(df)
        
        # 최고점 대비 하락폭 계산
        high_since_entry = df['high'].iloc[-int(hold_time/5):].max()
        drawdown_from_high = (current_price / high_since_entry - 1)
        
        # 1. 부분 익절 (목표의 60% 도달 시)
        if profit_ratio >= self.take_profit * self.partial_tp_ratio:
            if rsi.iloc[-1] > 70 or hist.iloc[-1] < 0:
                return True, "부분 목표 도달 익절"
                
        # 2. 트레일링 스탑
        if profit_ratio > self.take_profit * self.partial_tp_ratio and drawdown_from_high <= -self.trailing_stop:
            return True, "트레일링 스탑 익절"
            
        # 3. 기본 손절
        if profit_ratio <= self.stop_loss:
            return True, "기본 손절"
            
        # 4. 추세 반전 체크
        if len(df) >= self.trend_window:
            recent_trend = df['close'].iloc[-self.trend_window:]
            if (recent_trend.is_monotonic_decreasing and 
                profit_ratio < 0 and 
                not self.is_uptrend(df)):
                return True, "추세 반전 손절"
        
        # 5. 시간 초과 시 조건부 청산
        if hold_time >= self.max_hold_time:
            # 홀딩 연장 조건 체크
            if (rsi.iloc[-1] < 70 and 
                hist.iloc[-1] > 0 and
                profit_ratio > 0):  # 수익 중일 때만 연장
                if hold_time < self.max_hold_time * self.extended_hold_multiplier:
                    return False, ""  # 홀딩 연장
            
            # 홀딩 연장 조건 미충족 시 청산
            if profit_ratio > 0:
                return True, "시간초과 익절"
            else:
                return True, "시간초과 손절"
            
        return False, ""

class SpikeStrategy(BaseStrategy):
    def __init__(self, spike_threshold=1.5, volume_window=15):
        super().__init__()
        self.spike_threshold = spike_threshold
        self.volume_window = volume_window
        self.take_profit = 0.025
        self.stop_loss = -0.008
        self.max_hold_time = 180
        self.min_hold_time = 10
        self.ma_period = 20
        self.min_price_increase = 0.0025
        self.partial_take_profit = 0.015
        self.trailing_stop = 0.004
        self.min_entry_interval = 1800

    def evaluate_entry(self, df: pd.DataFrame) -> bool:
        if df is None or len(df) < self.volume_window + 1:
            return False
            
        df = df.copy()
        
        # 기본 지표 계산
        avg_volume = df['volume'].rolling(window=self.volume_window).mean()
        volume_spike = df['volume'].iloc[-1] > avg_volume.iloc[-1] * self.spike_threshold
        
        # 이동평균
        ma = df['close'].rolling(window=self.ma_period).mean()
        
        # 가격 상승률 계산
        price_increase = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1)
        prev_price_increase = (df['close'].iloc[-2] / df['close'].iloc[-3] - 1)
        
        # 기본 조건
        conditions = [
            volume_spike,                    # 거래량 급증 (임계값 상향)
            df['close'].iloc[-1] > df['open'].iloc[-1],  # 양봉
            df['close'].iloc[-1] > ma.iloc[-1],  # 단기 추세상승
            price_increase > self.min_price_increase,  # 최소 가격 상승률
            prev_price_increase > 0,  # 이전 캔들도 상승
            df['volume'].iloc[-1] > df['volume'].iloc[-2]  # 거래량 증가 추세
        ]
        
        return all(conditions)

    def evaluate_exit(self, df: pd.DataFrame, entry_price: float, hold_time: float) -> Tuple[bool, str]:
        if df is None or df.empty or hold_time < self.min_hold_time:
            return False, ""
            
        current_price = df.iloc[-1]['close']
        profit_ratio = self.calculate_profit_ratio(current_price, entry_price)
        
        # 최고점 대비 하락폭 계산
        high_since_entry = df['high'].iloc[-int(hold_time/5):].max()
        drawdown_from_high = (current_price / high_since_entry - 1)
        
        # 1. 기본 익절
        if profit_ratio >= self.take_profit:
            return True, "목표 익절"
            
        # 2. 부분 익절 (강세 지속 시)
        if profit_ratio >= self.partial_take_profit:
            momentum = df['close'].iloc[-1] / df['close'].iloc[-5] - 1
            if momentum < 0:  # 모멘텀 약화 시 익절
                return True, "모멘텀 약화로 익절"
                
        # 3. 트레일링 스탑
        if profit_ratio > self.partial_take_profit and drawdown_from_high <= -self.trailing_stop:
            return True, "트레일링 스탑 익절"
            
        # 4. 기본 손절
        if profit_ratio <= self.stop_loss:
            return True, "기본 손절"
            
        # 5. 시간 초과 시 조건부 청산
        if hold_time >= self.max_hold_time:
            if profit_ratio > 0:
                return True, "시간초과 익절"
            else:
                return True, "시간초과 손절"
            
        return False, ""

class VolatilityMALiteStrategy(BaseStrategy):
    """단순화된 변동성 전략"""
    def __init__(self):
        super().__init__()
        self.k = 0.5  # 변동성 계수 상향
        self.ma_period = 20
        self.take_profit = 0.025  # 익절 상향
        self.stop_loss = -0.015   # 손절 완화
        self.max_hold_time = 360  # 홀딩 시간 연장
        self.min_hold_time = 20   # 최소 홀딩 시간 증가
        self.volume_ma = 20
        self.min_volume_ratio = 1.2  # 거래량 기준 강화
        self.trailing_stop = 0.004
        self.min_entry_interval = 3600  # 진입 간격 확대 (1시간)
        self.trend_strength_threshold = 0.0002  # 추세 강도 기준

    def evaluate_entry(self, df: pd.DataFrame) -> bool:
        if df is None or len(df) < self.ma_period + 1:
            return False
            
        df = df.copy()
        
        # 기본 지표 계산
        df['ma'] = df['close'].rolling(window=self.ma_period).mean()
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # 추세 강도 계산
        ma_slope = (df['ma'].iloc[-1] - df['ma'].iloc[-2]) / df['ma'].iloc[-2]
        
        # 변동성 돌파 계산
        yesterday_range = df['high'].iloc[-2] - df['low'].iloc[-2]
        target_price = df['open'].iloc[-1] + yesterday_range * self.k
        
        # 개선된 진입 조건
        conditions = [
            df['close'].iloc[-1] > target_price,  # 변동성 돌파
            df['close'].iloc[-1] > df['ma'].iloc[-1],  # 이동평균선 위
            df['volume_ratio'].iloc[-1] > self.min_volume_ratio,  # 거래량 증가
            df['close'].iloc[-1] > df['open'].iloc[-1],  # 현재 양봉
            df['close'].iloc[-2] > df['open'].iloc[-2],  # 이전 양봉
            ma_slope > self.trend_strength_threshold,  # 상승 추세
            df['volume'].iloc[-1] > df['volume'].iloc[-2]  # 거래량 증가 추세
        ]
        
        return all(conditions)

    def evaluate_exit(self, df: pd.DataFrame, entry_price: float, hold_time: float) -> Tuple[bool, str]:
        if df is None or df.empty or hold_time < self.min_hold_time:
            return False, ""
            
        current_price = df.iloc[-1]['close']
        profit_ratio = self.calculate_profit_ratio(current_price, entry_price)
        
        # 최고점 대비 하락폭
        high_since_entry = df['high'].iloc[-int(hold_time/5):].max()
        drawdown_from_high = (current_price / high_since_entry - 1)
        
        # 이동평균 기준 하락 여부
        ma = df['close'].rolling(window=self.ma_period).mean()
        below_ma = current_price < ma.iloc[-1]
        
        # 개선된 청산 조건
        if profit_ratio >= self.take_profit:
            return True, "목표 익절"
            
        if profit_ratio <= self.stop_loss:
            return True, "손절"
            
        if drawdown_from_high <= -self.trailing_stop and below_ma:
            return True, "트레일링 스탑 + MA 하향"
            
        if hold_time >= self.max_hold_time:
            if profit_ratio > 0:
                return True, "시간초과 익절"
            elif below_ma:
                return True, "시간초과 손절"
            
        return False, ""

class RSIReversalStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.rsi_threshold = 30
        self.take_profit = 0.015
        self.stop_loss = -0.01
        self.min_hold_time = 10
        self.max_hold_time = 180
        self.rsi_recovery_level = 50
        self.min_entry_interval = 1800  # 30분 진입 제한

    def evaluate_entry(self, df: pd.DataFrame) -> bool:
        if df is None or len(df) < self.rsi_window + 2:
            return False
        df = df.copy()
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=self.rsi_window).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        last = df.iloc[-1]
        prev = df.iloc[-2]

        return (
            prev['rsi'] < self.rsi_threshold and
            last['rsi'] > prev['rsi'] and
            last['close'] > prev['close']
        )

    def evaluate_exit(self, df: pd.DataFrame, entry_price: float, hold_time: float) -> Tuple[bool, str]:
        if df is None or df.empty or hold_time < self.min_hold_time:
            return False, ""

        current_price = df.iloc[-1]['close']
        profit_ratio = self.calculate_profit_ratio(current_price, entry_price)

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=self.rsi_window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        if profit_ratio >= self.take_profit or rsi.iloc[-1] >= self.rsi_recovery_level:
            return True, "익절"

        if profit_ratio <= self.stop_loss:
            return True, "손절"

        if hold_time >= self.max_hold_time:
            if profit_ratio > 0:
                return True, "시간초과 익절"
            else:
                return True, "시간초과 손절"

        return False, ""

class CompositeStrategy(BaseStrategy):
    def __init__(self, strategies: List[BaseStrategy], weights: Dict[str, float]):
        super().__init__()
        self.strategies = strategies
        self.weights = weights
        self.position_manager = PositionManager()
        
    def evaluate_entry(self, df: pd.DataFrame) -> Tuple[bool, str, float]:
        for strategy in sorted(self.strategies, 
                             key=lambda x: self.position_manager.strategy_priorities[x.__class__.__name__]):
            if (strategy.evaluate_entry(df) and 
                self.position_manager.can_enter(strategy.__class__.__name__)):
                position_size = self.position_manager.get_position_size(
                    strategy.__class__.__name__, 
                    1.0,  # 기본 크기
                    self.weights
                )
                return True, strategy.__class__.__name__, position_size
        return False, "", 0.0

    def evaluate_exit(self, df: pd.DataFrame, entry_price: float, hold_time: float) -> Tuple[bool, str]:
        if self.position_manager.current_position is None:
            return False, ""
            
        for strategy in self.strategies:
            if (strategy.__class__.__name__ == self.position_manager.current_position):
                should_exit, reason = strategy.evaluate_exit(df, entry_price, hold_time)
                if should_exit:
                    self.position_manager.exit_position()
                return should_exit, reason
        
        return False, ""

# 전체 데이터 수집 함수 추가

def fetch_full_data(ticker: str, start_date: str, end_date: str, interval="minute5") -> pd.DataFrame:
    df_all = pd.DataFrame()
    to = datetime.strptime(end_date, "%Y-%m-%d")
    from_date = datetime.strptime(start_date, "%Y-%m-%d")
    
    logging.info(f"데이터 수집 시작: {start_date} ~ {end_date}")
    
    while True:
        logging.info(f"데이터 요청 중: {to.strftime('%Y-%m-%d %H:%M:%S')}")
        df = pyupbit.get_ohlcv(ticker, interval=interval, count=200, to=to.strftime("%Y-%m-%d %H:%M:%S"))
        if df is None or df.empty:
            break
        df_all = pd.concat([df, df_all])
        to = df.index[0] - timedelta(minutes=5)
        if df_all.index[0].date() <= from_date.date():
            break
        time.sleep(0.2)  # API 호출 간격 늘림
        
    logging.info(f"데이터 수집 완료: 총 {len(df_all)}개 데이터")
    return df_all[df_all.index.date >= from_date.date()]

# --- 백테스트용 메인 루프 ---
def run_backtest(strategy: BaseStrategy, ticker: str, start_date: str, end_date: str):
    df = fetch_full_data(ticker, start_date, end_date)
    if df is None or df.empty:
        logging.error("데이터를 불러오지 못했습니다.")
        return

    entry_price = None
    entry_time = None
    last_entry_time = None  # 마지막 진입 시간 추적
    holding = False
    trades = []
    position_size = 1.0

    for i in range(len(df)):
        current_df = df.iloc[:i+1]
        if len(current_df) < 20:
            continue

        current_time = current_df.index[-1]

        if not holding:
            # 진입 간격 체크
            can_enter = True
            if (last_entry_time is not None and 
                hasattr(strategy, 'min_entry_interval')):
                time_since_last_entry = (current_time - last_entry_time).total_seconds()
                can_enter = time_since_last_entry > strategy.min_entry_interval

            if can_enter:
                if isinstance(strategy, CompositeStrategy):
                    should_enter, strategy_name, size = strategy.evaluate_entry(current_df)
                    if should_enter:
                        position_size = size
                        strategy.position_manager.enter_position(strategy_name)
                        entry_price = current_df.iloc[-1]['close']
                        entry_time = current_time
                        last_entry_time = current_time
                        logging.info(f"[매수] {entry_time} @ {entry_price:,.0f} (크기: {position_size:.2f})")
                        holding = True
                elif strategy.evaluate_entry(current_df):
                    entry_price = current_df.iloc[-1]['close']
                    entry_time = current_time
                    last_entry_time = current_time
                    logging.info(f"[매수] {entry_time} @ {entry_price:,.0f}")
                    holding = True

        elif holding:
            hold_time = (current_time - entry_time).total_seconds() / 60
            should_exit, exit_reason = strategy.evaluate_exit(current_df, entry_price, hold_time)
            
            if should_exit:
                exit_price = current_df.iloc[-1]['close']
                pnl = (exit_price / entry_price - 1) * 100 * position_size
                logging.info(f"[매도] {current_time} @ {exit_price:,.0f} | 수익률: {pnl:.2f}% | 이유: {exit_reason}")
                trades.append({
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_time": current_time,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "reason": exit_reason,
                    "hold_time": hold_time,
                    "position_size": position_size
                })
                holding = False
                position_size = 1.0

    # 결과 출력
    df_trades = pd.DataFrame(trades)
    if not df_trades.empty:
        analyzer = BacktestAnalyzer(df_trades, strategy.__class__.__name__)
        analyzer.analyze()
    else:
        logging.info("거래 없음")
        
    return df_trades

class BacktestAnalyzer:
    def __init__(self, trades_df: pd.DataFrame, strategy_name: str):
        self.df = trades_df
        self.strategy_name = strategy_name
        self.prepare_data()
        self.log_file = "logs/backtest_summary.txt"  # 종합 분석 결과 저장 파일

    def prepare_data(self):
        if len(self.df) == 0:
            logging.warning(f"[{self.strategy_name}] 거래 기록이 없습니다.")
            return
        
        self.df['hold_time'] = (pd.to_datetime(self.df['exit_time']) - 
                               pd.to_datetime(self.df['entry_time'])).dt.total_seconds() / 60
        self.df['date'] = pd.to_datetime(self.df['entry_time']).dt.date

    def analyze(self):
        if len(self.df) == 0:
            metrics = self._empty_analysis()
        else:
            metrics = {
                'strategy': self.strategy_name,
                'total_trades': len(self.df),
                'win_rate': len(self.df[self.df['pnl'] > 0]) / len(self.df) * 100,
                'total_return': self.df['pnl'].sum(),
                'avg_return': self.df['pnl'].mean(),
                'std_return': self.df['pnl'].std(),
                'max_drawdown': self._calculate_max_drawdown(),
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'avg_hold_time': self.df['hold_time'].mean(),
                'risk_reward_ratio': abs(self.df[self.df['pnl'] > 0]['pnl'].mean() / 
                                    self.df[self.df['pnl'] < 0]['pnl'].mean() if len(self.df[self.df['pnl'] < 0]) > 0 else 0)
            }
        
        self._print_analysis(metrics)
        self._save_analysis(metrics)
        return metrics

    def _empty_analysis(self):
        return {
            'strategy': self.strategy_name,
            'total_trades': 0,
            'win_rate': 0.0,
            'total_return': 0.0,
            'avg_return': 0.0,
            'std_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'avg_hold_time': 0.0,
            'risk_reward_ratio': 0.0
        }

    def _calculate_max_drawdown(self):
        if len(self.df) == 0:
            return 0.0
        cumulative = (1 + self.df['pnl'] / 100).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        return drawdowns.min() * 100

    def _calculate_sharpe_ratio(self):
        if len(self.df) == 0 or self.df['pnl'].std() == 0:
            return 0.0
        return np.sqrt(252) * (self.df['pnl'].mean() / self.df['pnl'].std())

    def _print_analysis(self, metrics):
        analysis_text = f"\n=== {self.strategy_name} 백테스트 분석 결과 ===\n"
        analysis_text += f"총 거래 횟수: {metrics['total_trades']}회\n"
        analysis_text += f"승률: {metrics['win_rate']:.2f}%\n"
        analysis_text += f"총 수익률: {metrics['total_return']:.2f}%\n"
        analysis_text += f"평균 수익률: {metrics['avg_return']:.2f}%\n"
        analysis_text += f"수익률 표준편차: {metrics['std_return']:.2f}%\n"
        analysis_text += f"최대 낙폭: {metrics['max_drawdown']:.2f}%\n"
        analysis_text += f"샤프 비율: {metrics['sharpe_ratio']:.2f}\n"
        analysis_text += f"평균 보유시간: {metrics['avg_hold_time']:.1f}분\n"
        analysis_text += f"손익비: {metrics['risk_reward_ratio']:.2f}\n"
        
        logging.info(analysis_text)
        
        # 종합 분석 파일에 추가
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(analysis_text + "\n")

    def _save_analysis(self, metrics):
        # CSV 파일로 저장
        df_metrics = pd.DataFrame([metrics])
        df_metrics.to_csv(f"logs/{self.strategy_name}_analysis.csv", index=False)
        
        # 거래 상세 내역 저장
        if len(self.df) > 0:
            self.df.to_csv(f"logs/{self.strategy_name}_trades.csv", index=False)

def run_strategy_tests():
    """각 전략 독립적으로 테스트"""
    start_date = "2025-04-10"
    end_date = "2025-04-15"
    ticker = "KRW-BTC"
    
    logging.info(f"\n{'='*50}\n백테스트 시작\n기간: {start_date} ~ {end_date}\n종목: {ticker}\n{'='*50}")
    
    # 기존 분석 결과 파일 초기화
    with open("logs/backtest_summary.txt", 'w', encoding='utf-8') as f:
        f.write(f"백테스트 실행 시간: {datetime.now()}\n")
        f.write(f"대상 기간: {start_date} ~ {end_date}\n")
        f.write(f"대상 종목: {ticker}\n\n")
    
    results = {}
    
    # 1. Spike 전략 단독 테스트
    spike_strategy = SpikeStrategy(spike_threshold=1.5)
    results['spike'] = run_backtest(
        strategy=spike_strategy,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date
    )
    
    # 2. VolatilityMA 전략 단독 테스트
    vol_strategy = VolatilityMAStrategy()
    results['volatility'] = run_backtest(
        strategy=vol_strategy,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date
    )
    
    # 3. VolatilityMA Lite 전략 테스트
    vol_lite_strategy = VolatilityMALiteStrategy()
    results['volatility_lite'] = run_backtest(
        strategy=vol_lite_strategy,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date
    )

    # 4. RSI Reversal 전략 테스트
    rsi_reversal_strategy = RSIReversalStrategy()
    results['rsi_reversal'] = run_backtest(
        strategy=rsi_reversal_strategy,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date
    )
    
    # 5. 복합 전략 테스트 (가중치 조정)
    composite_or = CompositeStrategy(
        strategies=[vol_strategy, spike_strategy, vol_lite_strategy, rsi_reversal_strategy],
        weights={
            'VolatilityMAStrategy': 0.4,    # 기본 변동성 전략
            'SpikeStrategy': 0.2,           # 스파이크 전략
            'VolatilityMALiteStrategy': 0.2, # 라이트 전략
            'RSIReversalStrategy': 0.2      # RSI 역추세 전략
        }
    )
    results['composite_or'] = run_backtest(
        strategy=composite_or,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date
    )
    
    # 전략 비교 분석
    compare_strategies(results)

def compare_strategies(results: Dict[str, pd.DataFrame]):
    """전략 간 성과 비교 분석"""
    comparison_text = "\n=== 전략 비교 분석 ===\n"
    
    # 각 전략의 주요 지표 비교
    metrics = {}
    for strategy_name, trades_df in results.items():
        if len(trades_df) > 0:
            metrics[strategy_name] = {
                'total_trades': len(trades_df),
                'win_rate': len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100,
                'total_return': trades_df['pnl'].sum(),
                'sharpe_ratio': np.sqrt(252) * (trades_df['pnl'].mean() / trades_df['pnl'].std()) if trades_df['pnl'].std() != 0 else 0
            }
        else:
            metrics[strategy_name] = {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'sharpe_ratio': 0
            }
    
    # 결과를 DataFrame으로 변환하여 보기 좋게 출력
    df_comparison = pd.DataFrame(metrics).T
    df_comparison.columns = ['총 거래수', '승률(%)', '총 수익률(%)', '샤프 비율']
    
    comparison_text += "\n" + str(df_comparison) + "\n\n"
    comparison_text += "최고 성과 전략:\n"
    comparison_text += f"- 승률 기준: {df_comparison['승률(%)'].idxmax()} ({df_comparison['승률(%)'].max():.2f}%)\n"
    comparison_text += f"- 수익률 기준: {df_comparison['총 수익률(%)'].idxmax()} ({df_comparison['총 수익률(%)'].max():.2f}%)\n"
    comparison_text += f"- 샤프 비율 기준: {df_comparison['샤프 비율'].idxmax()} ({df_comparison['샤프 비율'].max():.2f})\n"
    
    # 결과를 파일에 저장
    with open("logs/backtest_summary.txt", 'a', encoding='utf-8') as f:
        f.write(comparison_text)
    
    logging.info(comparison_text)

if __name__ == '__main__':
    run_strategy_tests()
