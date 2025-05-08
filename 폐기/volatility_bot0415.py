#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import logging
import logging.handlers
import pyupbit
import signal
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from colorama import init, Fore, Style
import numpy as np
import pandas as pd
import csv
import glob
import zipfile
import argparse
from enum import Enum

# 터미널 색상 초기화 (자동 리셋)
init(autoreset=True)

def setup_logging():
    """로깅 시스템 설정"""
    # 기존 핸들러 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 로거 생성
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 루트 로거 레벨 설정
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 콘솔 핸들러 레벨을 INFO로 설정
    
    class ColoredFormatter(logging.Formatter):
        def format(self, record):
            # 로그 레벨에 따른 색상 설정
            if record.levelno >= logging.ERROR:
                level_color = Fore.RED
                msg_color = Fore.RED
            elif record.levelno >= logging.WARNING:
                level_color = Fore.YELLOW
                msg_color = Style.RESET_ALL
            else:
                level_color = Fore.BLUE
                msg_color = Style.RESET_ALL
            
            # 시간 색상은 항상 초록색
            record.colored_time = f"{Fore.GREEN}{self.formatTime(record, self.datefmt)}{Style.RESET_ALL}"
            record.colored_levelname = f"{level_color}{record.levelname}{Style.RESET_ALL}"
            record.colored_message = f"{msg_color}{record.getMessage()}{Style.RESET_ALL}"
            
            return f"{record.colored_time} [{record.colored_levelname}] {record.colored_message}"
    
    console_formatter = ColoredFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (로테이션 적용)
    file_handler = logging.handlers.RotatingFileHandler(
        filename="trade_log.txt",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)  # 파일에는 모든 로그 기록
    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

# ===============================
# 1. 기본 설정 및 로깅 구성
# ===============================

# 로깅 시스템 초기화
logger = setup_logging()

def safe_api_call(func, *args, **kwargs):
    """API 호출을 안전하게 처리하는 래퍼 함수"""
    max_retries = 3
    retry_delay = 1  # 초
    last_error = None

    for attempt in range(max_retries):
        try:
            # API 호출 전 대기
            if attempt > 0:
                time.sleep(retry_delay * (attempt + 1))  # 재시도마다 대기 시간 증가
            
            result = func(*args, **kwargs)
            
            # DataFrame 결과 처리
            if isinstance(result, pd.DataFrame):
                if result.empty:
                    last_error = "빈 DataFrame"
                    continue
                return result
            
            # 숫자형 결과 처리
            if isinstance(result, (int, float)):
                if result == 0:
                    last_error = "결과값 0"
                    continue
                return result
            
            # 딕셔너리/리스트 결과 처리
            if isinstance(result, (dict, list)):
                if not result:  # 빈 딕셔너리/리스트 체크
                    last_error = "빈 데이터"
                    continue
                return result
            
            # None 체크
            if result is None:
                last_error = "None 결과"
                continue
            
            # 기타 유효한 결과
            return result
            
        except Exception as e:
            last_error = str(e)
            if "429" in str(e):  # Too Many Requests
                time.sleep(retry_delay * 2)
            elif "500" in str(e):  # Server Error
                time.sleep(retry_delay * 3)
            continue

    # 최종 실패 시 상세 로그
    logger.error(f"API 호출 최종 실패 - 마지막 오류: {last_error}")
    return None

def get_current_price_safely(ticker):
    """현재가 조회 함수"""
    try:
        # 우선 현재가 API 시도
        price = safe_api_call(pyupbit.get_current_price, ticker)
        if price is not None and price > 0:
            return price
            
        # 실패 시 최근 체결 내역으로 시도
        trades = safe_api_call(pyupbit.get_trades, ticker, limit=1)
        if trades and isinstance(trades, list) and len(trades) > 0:
            return float(trades[0]['trade_price'])
            
        # 모두 실패 시 주문서 현재가 조회
        orderbook = safe_api_call(pyupbit.get_orderbook, ticker)
        if orderbook and 'orderbook_units' in orderbook and len(orderbook['orderbook_units']) > 0:
            return float(orderbook['orderbook_units'][0]['ask_price'])
            
        return None
    except Exception as e:
        logger.error(f"{ticker} 현재가 조회 중 오류: {str(e)}")
        return None

# 환경변수 로드 및 API 키 설정
load_dotenv()
UPBIT_ACCESS = os.getenv("ACCESS_KEY")
UPBIT_SECRET = os.getenv("SECRET_KEY")

if not UPBIT_ACCESS or not UPBIT_SECRET:
    logger.error("API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    raise ValueError("API 키 오류")

try:
    upbit = pyupbit.Upbit(UPBIT_ACCESS, UPBIT_SECRET)
    balance = safe_api_call(upbit.get_balance, "KRW")
    if balance is None:
        logger.error("잔고 조회 실패. API 키를 확인해주세요.")
        raise ValueError("API 키 오류")
    logger.info(f"초기 KRW 잔고: {balance:,.0f}원")
except Exception as e:
    logger.error(f"Upbit 초기화 실패: {str(e)}")
    raise

# -------------------------------
# 설정값 클래스 (상단에 정의되어 쉽게 변경 가능)
# -------------------------------
class Settings:
    class Trading:
        # 기본 거래 설정
        BASE_TRADE_AMOUNT = 300000      # 기본 거래 금액
        MIN_TRADE_AMOUNT = 200000       # 최소 거래 금액
        COMMISSION_RATE = 0.005         # 거래 수수료 비율
        MAX_HOLDINGS = 3                # 최대 보유 코인 수
        
        # 매매 전략 설정
        CONFIDENCE_THRESHOLD = 75       # 매수 신뢰도 기준 강화
        TAKE_PROFIT_PERCENT = 1.2       # 익절 목표 하향
        STOP_LOSS_PERCENT = 0.6         # 손절 기준 강화
        FORCE_SELL_MINUTES = 20         # 강제 청산 시간 단축
        MIN_HOLDING_MINUTES = 5         # 최소 홀딩 시간

        # 추가 매수 설정
        ENABLE_ADDITIONAL_BUY = True    # 추가 매수 활성화
        MAX_BUYS_PER_COIN = 2          # 코인당 최대 매수 횟수
        ADDITIONAL_BUY_DIP = 0.5        # 추가 매수 가격 하락 기준
        MAX_POSITION_SIZE = 400000      # 코인당 최대 투자금액

    class Technical:
        # RSI 설정
        RSI_PERIOD = 14
        RSI_OVERSOLD = 30              # RSI 과매도 기준 강화
        RSI_WEIGHT = 0.3               # RSI 가중치

        # 볼린저 밴드 설정
        BB_WINDOW = 20
        BB_STD_DEV = 2
        BB_WEIGHT = 0.2                # BB 가중치 추가

        # 거래량 급등 전략
        VOLUME_BREAKOUT_WEIGHT = 0.3    # 거래량 가중치 조정
        MIN_VOLUME_RATIO = 2.0          # 거래량 급등 기준 강화
        MIN_MOMENTUM = 0.3              # 모멘텀 요구치 조정
        
        # 이동평균선 설정
        MA_WEIGHT = 0.2                 # MA 가중치
        MA_FAST = 5                     # 단기 이동평균
        MA_SLOW = 20                    # 장기 이동평균

    class Market:
        # 거래량 필터링 (상향 조정)
        MIN_TRADE_VOLUME = 200000       # 최소 거래량 조건 상향
        VOLUME_UPDATE_INTERVAL = 15     # 거래량 필터링 주기 단축
        
        # 제외 코인 (스테이블 코인 + 고위험 코인)
        STABLE_COINS = [
            # 스테이블 코인
            "KRW-USDT", "KRW-DAI", "KRW-TUSD", "KRW-BUSD", "KRW-USDC",
            # 고변동성 코인
            "KRW-DOGE", "KRW-XRP", "KRW-BTT", "KRW-WAXP",
            # 뉴스/SNS 민감 코인
            "KRW-SAND", "KRW-MANA", "KRW-SHIB", "KRW-APT"
        ]  # 고위험 코인 추가

    class System:
        # API 요청 관리
        API_REQUEST_INTERVAL = 0.5      # API 요청 간격 (0.1초에서 0.5초로 증가)
        API_ERROR_INTERVAL = 2.0        # API 오류 발생 시 대기 시간
        CACHE_EXPIRY = 300             # 캐시 만료 시간 (초)
        
        # 로그 설정
        LOG_ROTATION_SIZE = 10485760    # 로그 파일 최대 크기 (10MB)
        LOG_BACKUP_COUNT = 5            # 보관할 로그 파일 수

class MarketCondition(Enum):
    NORMAL = "normal"
    VOLATILE = "volatile"
    TRENDING = "trending"
    SIDEWAYS = "sideways"
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"

class MarketVolatility(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class MarketVolume(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    BREAKOUT = "breakout"

# 캐시 관리
class Cache:
    def __init__(self):
        self.data = {}
        
    def get(self, key):
        if key in self.data:
            value, timestamp = self.data[key]
            if time.time() - timestamp < Settings.System.CACHE_EXPIRY:
                return value
            del self.data[key]
        return None
        
    def set(self, key, value):
        self.data[key] = (value, time.time())
        
    def clear(self):
        self.data.clear()

# 캐시 인스턴스 생성
ohlcv_cache = Cache()
indicator_cache = Cache()

# ===============================
# 전역 변수: 보유 코인 및 거래 기록
# ===============================
held_coins = {}  # {ticker: {"buy_price": float, "buy_time": datetime}}
trade_history = []  # 거래 내역을 JSON 파일에 저장할 예정
trading_logic = None  # TradingLogic 인스턴스를 저장할 전역 변수
data_manager = None   # DataManager 인스턴스를 저장할 전역 변수

# 사이클 및 누적 카운터
cycle_count = 0
cumulative_counts = {
    'buy': 0,
    'tp': 0,
    'sl_normal': 0,  # 일반 손절
    'sl_forced': 0,  # 강제 청산
    'sl_other': 0,   # 기타 손절
    'error': 0
}
cumulative_profit = 1.0

# 전역 변수 설정
emergency_stop = False  # 긴급 종료 플래그

# -------------------------------
# 2. 시간대 설정 (한국 시간 기준)
# -------------------------------
KST = ZoneInfo("Asia/Seoul")

# -------------------------------
# 3. 코인 필터링: KRW 전체 마켓 조회 및 스테이블 코인 제외
# -------------------------------
def load_markets():
    all_tickers = pyupbit.get_tickers(fiat="KRW")
    market_list = [ticker for ticker in all_tickers if ticker not in Settings.Market.STABLE_COINS]
    logger.info(f"전체 KRW 마켓 조회, 대상 코인 수: {len(market_list)}")
    return market_list

def filter_by_volume(markets):
    """거래량 기준 필터링"""
    filtered = []
    volume_cache = {}  # 거래량 캐시 추가
    
    for ticker in markets:
        # 캐시된 거래량 확인
        if ticker in volume_cache:
            avg_volume = volume_cache[ticker]
        else:
            df = safe_api_call(pyupbit.get_ohlcv, ticker, interval="minute5", count=12)  # 1시간 데이터만 조회
            if df is None or df.empty:
                continue
            avg_volume = df['volume'].mean()
            volume_cache[ticker] = avg_volume
            
        if avg_volume >= Settings.Market.MIN_TRADE_VOLUME:
            filtered.append(ticker)
            
    logger.info(f"거래량 기준 필터링 후 대상 코인 수: {len(filtered)}")
    return filtered

# -------------------------------
# 4. 시장 상태 분석 및 가중치 적용
# -------------------------------
def analyze_market_state(tickers):
    """전체 시장의 평균 RSI를 기준으로 시장 상태(상승장/일반장/하락장)를 판단"""
    if not tickers:  # 빈 리스트 체크 추가
        logger.warning("분석할 코인이 없습니다.")
        return 1.0  # 기본 가중치 반환

    rsi_values = []
    for ticker in tickers[:10]:  # 상위 10개 코인만 분석 (API 부하 감소)
        try:
            df = safe_api_call(pyupbit.get_ohlcv, ticker, interval="day", count=15)
            time.sleep(Settings.System.API_REQUEST_INTERVAL)
            
            if df is not None and not df.empty and len(df) >= Settings.Technical.RSI_PERIOD:
                closes = df['close'].values
                deltas = np.diff(closes)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                
                avg_gain = np.mean(gains[:Settings.Technical.RSI_PERIOD])
                avg_loss = np.mean(losses[:Settings.Technical.RSI_PERIOD])
                
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    
                rsi_values.append(rsi)
                logger.debug(f"{ticker} RSI: {rsi:.2f}")
        except Exception as e:
            logger.error(f"{ticker} 시장 상태 분석 오류: {e}")
            continue

    if not rsi_values:  # RSI 계산 실패 시
        logger.warning("RSI 계산 실패. 기본 가중치를 사용합니다.")
        return 1.0

    avg_rsi = np.mean(rsi_values)
    logger.info(f"전체 시장 평균 RSI: {avg_rsi:.2f}")

    # 상승장, 일반장, 하락장에 따른 가중치 설정
    if avg_rsi > 60:
        weight = 1.2
        state = "상승장"
    elif avg_rsi < 40:
        weight = 0.8
        state = "하락장"
    else:
        weight = 1.0
        state = "일반장"

    logger.info(f"시장 상태: {state} (가중치: {weight})")
    return weight

# -------------------------------
# 5. 신뢰도 기반 매수 로직
# -------------------------------
def calculate_rsi(df, period):
    """RSI 계산 함수 수정"""
    if len(df) < period + 1:
        return None
    
    # close 가격의 변화
    delta = df['close'].diff()
    
    # 상승폭과 하락폭 구분
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # 평균 계산
    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()
    
    # RSI 계산
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1]  # 최신 RSI 값만 반환

def calculate_bb_position(df, window, std_dev):
    """볼린저 밴드 포지션 계산 함수 수정"""
    if len(df) < window:
        return None
        
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    
    if pd.isna(rolling_mean.iloc[-1]) or pd.isna(rolling_std.iloc[-1]):
        return None
        
    upper_band = rolling_mean + std_dev * rolling_std
    lower_band = rolling_mean - std_dev * rolling_std
    
    current_price = df['close'].iloc[-1]
    
    # 밴드 폭이 0인 경우 처리
    band_width = upper_band.iloc[-1] - lower_band.iloc[-1]
    if band_width == 0:
        return 50.0
        
    position = (current_price - lower_band.iloc[-1]) / band_width
    return position * 100  # 백분율로 변환

def calculate_confidence(ticker):
    """
    개별 코인의 매수 신뢰도를 계산
    - RSI/BB 전략 (60%): RSI 값과 볼린저 밴드 위치 기반
    - 변동성 돌파 전략 (40%): 전일 변동성과 현재가 기반
    """
    df = safe_api_call(pyupbit.get_ohlcv, ticker, interval="minute5", count=100)
    if df is None or df.empty or len(df) < Settings.Technical.BB_WINDOW:
        return 0, None, None
        
    try:
        # 1. RSI/BB 전략 신뢰도 계산 (60%)
        rsi = calculate_rsi(df, Settings.Technical.RSI_PERIOD)
        if rsi is None:
            return 0, None, None
            
        # RSI 신뢰도: 과매도 구간에서 높은 점수
        if rsi <= 30:
            rsi_conf = 90 + (30 - rsi)  # 20-30: 90-100점
        elif rsi <= 40:
            rsi_conf = 70 + (40 - rsi) * 2  # 30-40: 70-90점
        elif rsi <= 50:
            rsi_conf = 50 + (50 - rsi) * 2  # 40-50: 50-70점
        elif rsi <= 60:
            rsi_conf = 30 + (60 - rsi) * 2  # 50-60: 30-50점
        elif rsi <= 70:
            rsi_conf = 10 + (70 - rsi) * 2  # 60-70: 10-30점
        else:
            rsi_conf = max(0, 100 - rsi)  # 70 이상: 0-10점
            
        # BB 위치에 따른 신뢰도
        bb_pos = calculate_bb_position(df, Settings.Technical.BB_WINDOW, Settings.Technical.BB_STD_DEV)
        if bb_pos is None:
            return 0, None, None
            
        if bb_pos <= 20:
            bb_conf = 80 + (20 - bb_pos)  # 0-20%: 80-100점
        elif bb_pos <= 40:
            bb_conf = 60 + (40 - bb_pos)  # 20-40%: 60-80점
        elif bb_pos <= 60:
            bb_conf = 40 + (60 - bb_pos) / 2  # 40-60%: 40-60점
        elif bb_pos <= 80:
            bb_conf = 20 + (80 - bb_pos) / 2  # 60-80%: 20-40점
        else:
            bb_conf = max(0, 100 - bb_pos)  # 80-100%: 0-20점
            
        # RSI/BB 전략 최종 신뢰도 (RSI와 BB 각각 30%)
        rsi_bb_conf = (rsi_conf * 0.5 + bb_conf * 0.5) * 0.6
        
        # 2. 변동성 돌파 전략 신뢰도 계산 (40%)
        daily_df = safe_api_call(pyupbit.get_ohlcv, ticker, interval="day", count=2)
        if daily_df is None or daily_df.empty or len(daily_df) < 2:
            return 0, None, None
            
        prev_day = daily_df.iloc[-2]
        today = daily_df.iloc[-1]
        
        # 변동성 계산 (전일 고가 - 전일 저가)
        volatility = prev_day['high'] - prev_day['low']
        k = Settings.Technical.VOLATILITY_K
        target_price = prev_day['close'] + volatility * k
        
        current_price = today['close']
        
        # 목표가 돌파 시 신뢰도 증가
        if current_price > target_price:
            volatility_conf = min(100, (current_price - target_price) / target_price * 1000)
        else:
            volatility_conf = 0
            
        volatility_conf = volatility_conf * 0.4  # 40% 비중
        
        # 최종 신뢰도 계산
        confidence = rsi_bb_conf + volatility_conf
        
        logger.info(f"{ticker} 신뢰도 분석:")
        logger.info(f"- RSI({rsi:.1f})/BB({bb_pos:.1f}%) 전략: {rsi_bb_conf:.1f}점")
        logger.info(f"- 변동성 돌파 전략: {volatility_conf:.1f}점")
        logger.info(f"- 최종 신뢰도: {confidence:.1f}점")
        
        return confidence, rsi, bb_pos
        
    except Exception as e:
        logger.error(f"{ticker} 신뢰도 계산 중 오류: {str(e)}")
        return 0, None, None

def should_buy(ticker, market_weight):
    """개별 코인의 종합 신뢰도가 기준 이상이면 매수 신호 생성"""
    confidence, rsi, bb_pos = calculate_confidence(ticker)
    overall_confidence = confidence * market_weight
    logger.info(f"{ticker} - 종합 신뢰도: {overall_confidence:.2f}% (기준: {Settings.Trading.CONFIDENCE_THRESHOLD}%)")
    return overall_confidence >= Settings.Trading.CONFIDENCE_THRESHOLD, overall_confidence

# -------------------------------
# 5-1. 매도 로직
# -------------------------------
class TradingLogic:
    def __init__(self, settings, data_manager):
        self.settings = settings
        self.atr_period = 14
        self.atr_multiplier_tp = 1.5
        self.atr_multiplier_sl = 0.7
        self.trailing_activation = 0.05
        self.trailing_stop_percent = 0.02
        self.held_coins = {}
        self.data_manager = data_manager
        self.strategy_composer = StrategyComposer()
        self.load_held_coins()

    def execute_trade_decision(self, ticker, current_price, buy_time=None, buy_price=None):
        """매매 결정 실행"""
        try:
            # 매도 체크
            if ticker in self.held_coins:
                should_sell, reason = self.should_sell(ticker, current_price)
                if should_sell:
                    return "SELL", reason
                return "HOLD", None

            # 매수 체크
            should_buy_result, confidence, buy_reasons = self.should_buy(ticker, current_price)
            if should_buy_result:
                return "BUY", confidence, buy_reasons
            return "HOLD", None

        except Exception as e:
            logger.error(f"매매 결정 실행 중 오류: {str(e)}")
            return "HOLD", None

    def should_sell(self, ticker, current_price):
        """매도 조건 확인"""
        if ticker not in self.held_coins:
            return False, "NOT_HELD"

        coin_data = self.held_coins[ticker]
        buy_price = coin_data['buy_price']
        buy_time = coin_data['buy_time']
        
        # 현재 수익률 계산
        current_profit = (current_price - buy_price) / buy_price * 100
        
        # 동적 목표가 계산
        take_profit, stop_loss = self.get_dynamic_targets(ticker, current_price)
        
        # 트레일링 스탑 업데이트
        self.update_trailing_stop(ticker, current_price)
        
        # 최소 보유 시간 확인
        holding_time = self.calculate_holding_time(ticker)
        if holding_time < self.settings.Trading.MIN_HOLDING_MINUTES:
            return False, "TOO_SHORT"

        # 강제 청산 시간 확인
        if holding_time >= self.settings.Trading.FORCE_SELL_MINUTES:
            # 수익중인 경우 추가 시간 부여
            if current_profit > 0:
                if holding_time < self.settings.Trading.FORCE_SELL_MINUTES * 1.5:  # 50% 더 기다림
                    return False, "PROFIT_EXTENSION"
            return True, "FORCE_SELL"

        # 익절 조건 확인
        if current_price >= take_profit:
            return True, "TAKE_PROFIT"

        # 손절 조건 확인
        if current_price <= stop_loss:
            # 추가 매수 가능 여부 확인
            if self.can_buy_more(ticker, current_price):
                return False, "ADDITIONAL_BUY"
            return True, "STOP_LOSS"

        return False, "HOLDING"

    def execute_sell(self, ticker, current_price, reason=''):
        """매도 실행"""
        if ticker not in self.held_coins:
            return False

        coin_data = self.held_coins[ticker]
        buy_price = coin_data['buy_price']
        profit_percent = (current_price - buy_price) / buy_price * 100
        
        try:
            # 분할 매도 여부 결정
            if profit_percent > 2.0 and reason == "TAKE_PROFIT":
                return self._execute_partial_sell(ticker, current_price, profit_percent)
            else:
                return self._execute_full_sell(ticker, current_price, reason)
        except Exception as e:
            logger.error(f"{ticker} 매도 실패: {str(e)}")
            return False

    def _execute_partial_sell(self, ticker, current_price, profit_percent):
        """분할 매도 실행"""
        coin_data = self.held_coins[ticker]
        
        # 수익률에 따른 매도 비율 결정
        if profit_percent > 5.0:
            sell_ratio = 0.7  # 70% 매도
        elif profit_percent > 3.0:
            sell_ratio = 0.5  # 50% 매도
        else:
            sell_ratio = 0.3  # 30% 매도

        try:
            # 매도할 수량 계산
            total_balance = self.upbit.get_balance(ticker)
            sell_balance = total_balance * sell_ratio
            
            # 매도 주문
            order = self.upbit.sell_market_order(ticker, sell_balance)
            if order and 'error' not in order:
                # 매도 정보 업데이트
                remaining_balance = total_balance - sell_balance
                if remaining_balance > 0:
                    coin_data['investment'] *= (1 - sell_ratio)
                    logger.info(f"{ticker} 부분 매도 성공: {sell_ratio*100:.0f}% (수익률: {profit_percent:.1f}%)")
                    return True
                else:
                    del self.held_coins[ticker]
                    logger.info(f"{ticker} 전체 매도 성공: (수익률: {profit_percent:.1f}%)")
                    return True
        except Exception as e:
            logger.error(f"{ticker} 부분 매도 실패: {str(e)}")
        
        return False

    def _execute_full_sell(self, ticker, current_price, reason):
        """전체 매도 실행"""
        try:
            # 전체 수량 매도
            order = self.upbit.sell_market_order(ticker, self.upbit.get_balance(ticker))
            if order and 'error' not in order:
                profit_percent = (current_price - self.held_coins[ticker]['buy_price']) / self.held_coins[ticker]['buy_price'] * 100
                logger.info(f"{ticker} 매도 성공: {reason} (수익률: {profit_percent:.1f}%)")
                del self.held_coins[ticker]
                return True
        except Exception as e:
            logger.error(f"{ticker} 매도 실패: {str(e)}")
        
        return False

    def update_trailing_stop(self, ticker, current_price):
        """트레일링 스탑 업데이트"""
        if ticker not in self.held_coins:
            return

        coin_data = self.held_coins[ticker]
        buy_price = coin_data['buy_price']
        
        # 현재 수익률 계산
        current_profit = (current_price - buy_price) / buy_price * 100
        
        # 트레일링 스탑 초기화
        if 'trailing_stop' not in coin_data:
            coin_data['trailing_stop'] = buy_price * (1 - self.settings.Trading.STOP_LOSS_PERCENT / 100)
            coin_data['highest_price'] = current_price
            return

        # 신규 고점 갱신 시 트레일링 스탑 상향 조정
        if current_price > coin_data['highest_price']:
            coin_data['highest_price'] = current_price
            # 수익률에 따른 트레일링 스탑 비율 조정
            if current_profit > 2.0:
                trail_ratio = 0.7  # 수익 70% 보호
            elif current_profit > 1.5:
                trail_ratio = 0.6  # 수익 60% 보호
            elif current_profit > 1.0:
                trail_ratio = 0.5  # 수익 50% 보호
            else:
                trail_ratio = 0.4  # 수익 40% 보호
            
            # 새로운 트레일링 스탑 설정
            new_stop = current_price * (1 - (1 - trail_ratio) * self.settings.Trading.STOP_LOSS_PERCENT / 100)
            coin_data['trailing_stop'] = max(new_stop, coin_data['trailing_stop'])

    def get_dynamic_targets(self, ticker, current_price):
        """동적 목표가 계산"""
        if ticker not in self.held_coins:
            return None, None

        coin_data = self.held_coins[ticker]
        buy_price = coin_data['buy_price']
        
        # ATR 기반 변동성 계산
        atr = self.calculate_atr(ticker)
        if atr is None:
            return None, None

        # 현재 수익률 계산
        current_profit = (current_price - buy_price) / buy_price * 100
        
        # 변동성과 수익률에 따른 동적 목표가 조정
        if current_profit > 2.0:
            tp_multiplier = 2.0  # 수익률이 높을 때는 목표가를 더 높게 설정
        elif current_profit > 1.0:
            tp_multiplier = 1.5
        else:
            tp_multiplier = 1.2

        # 목표가 = 현재가 + (ATR * 승수)
        take_profit = current_price + (atr * tp_multiplier)
        
        # 손절가 = 트레일링 스탑 또는 기본 손절가
        if 'trailing_stop' in coin_data:
            stop_loss = coin_data['trailing_stop']
        else:
            stop_loss = current_price - (atr * self.settings.Technical.ATR_MULTIPLIER_SL)

        return take_profit, stop_loss

    def calculate_atr(self, ticker):
        """ATR(Average True Range) 계산"""
        try:
            df = pyupbit.get_ohlcv(ticker, interval="minute60", count=self.atr_period+1)
            if df is None or len(df) < self.atr_period:
                logger.error(f"{ticker} ATR 계산을 위한 데이터 부족")
                return None

            high = df['high']
            low = df['low']
            close = df['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=self.atr_period).mean()
            
            return atr.iloc[-1]
        except Exception as e:
            logger.error(f"{ticker} ATR 계산 중 오류: {str(e)}")
            return None

    def calculate_holding_time(self, ticker):
        """보유 시간 계산 (분)"""
        if ticker not in self.held_coins:
            return 0
        buy_time = self.held_coins[ticker]["buy_time"]
        holding_time = datetime.now(KST) - buy_time
        return holding_time.total_seconds() / 60

    def get_market_condition(self, ticker):
        """시장 상태 판단"""
        try:
            df = pyupbit.get_ohlcv(ticker, interval="minute60", count=24)
            if df is None:
                return "UNKNOWN"
            
            price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
            
            if price_change > 5:
                return "STRONG_UP"
            elif price_change > 2:
                return "UP"
            elif price_change < -5:
                return "STRONG_DOWN"
            elif price_change < -2:
                return "DOWN"
            else:
                return "SIDEWAYS"
                
        except Exception as e:
            logger.error(f"시장 상태 확인 중 오류: {str(e)}")
            return "UNKNOWN"

    def load_held_coins(self):
        """보유 중인 코인 정보 로드"""
        try:
            balances = safe_api_call(upbit.get_balances)
            if not balances:
                logger.info("보유 중인 코인이 없습니다.")
                return

            for balance in balances:
                if balance['currency'] != 'KRW':
                    ticker = f"KRW-{balance['currency']}"
                    amount = float(balance['balance'])
                    avg_buy_price = float(balance['avg_buy_price'])
                    
                    self.held_coins[ticker] = {
                        'amount': amount,
                        'buy_price': avg_buy_price,
                        'buy_time': datetime.now(KST),  # 실제 매수 시간은 알 수 없으므로 현재 시간으로 설정
                        'confidence': 0  # 기본값
                    }
                    
            logger.info(f"보유 코인 로드 완료: {list(self.held_coins.keys())}")
            
        except Exception as e:
            logger.error(f"보유 코인 정보 로드 중 오류: {str(e)}")

    def should_buy(self, ticker, current_price):
        """매수 여부 결정"""
        try:
            # 이미 보유 중인 코인인 경우 추가 매수 여부 확인
            if ticker in self.held_coins:
                buy_price = self.held_coins[ticker]["buy_price"]
                # 현재가가 매수 평균가보다 높으면 추가 매수하지 않음
                if current_price >= buy_price:
                    return False, 0, None
            
            # OHLCV 데이터 조회
            df = get_ohlcv_cached(ticker, interval="minute5", count=100)
            if df is None or df.empty or len(df) < Settings.Technical.BB_WINDOW:
                logger.debug(f"{ticker} - 충분한 OHLCV 데이터를 얻지 못함")
                return False, 0, None
            
            # 전략 분석 실행
            analysis = self.strategy_composer.analyze(ticker, df, current_price)
            if analysis is None:
                logger.debug(f"{ticker} - 전략 분석 실패")
                return False, 0, None
            
            total_confidence = analysis.get('total_confidence', 0)
            
            # 매수 이유 생성
            buy_reasons = self._generate_buy_reasons(analysis, current_price)
            
            # 신뢰도가 기준값 이상이면 매수 신호
            return total_confidence >= Settings.Trading.CONFIDENCE_THRESHOLD, total_confidence, buy_reasons
            
        except Exception as e:
            logger.error(f"{ticker} 매수 신호 계산 중 오류: {str(e)}")
            return False, 0, None

    def _generate_buy_reasons(self, analysis, current_price):
        """매수 이유 생성"""
        try:
            reasons = []
            results = analysis.get('results', {})
            
            # RSI/BB 전략 분석
            if 'RSI_BB' in results:
                rsi_bb = results['RSI_BB']
                rsi = rsi_bb.get('rsi', 0)
                bb_pos = rsi_bb.get('bb_position', 0)
                
                if rsi <= 30:
                    reasons.append(f"RSI 과매도({rsi:.1f})")
                elif rsi <= 40:
                    reasons.append(f"RSI 매수구간({rsi:.1f})")
                    
                if bb_pos <= 20:
                    reasons.append(f"BB 하단 접근({bb_pos:.1f}%)")
                elif bb_pos <= 40:
                    reasons.append(f"BB 매수구간({bb_pos:.1f}%)")
            
            # 변동성 돌파 전략 분석
            if 'VOLATILITY' in results:
                vol = results['VOLATILITY']
                target = vol.get('target_price', 0)
                volatility = vol.get('volatility', 0)
                
                if current_price > target:
                    breakout_ratio = ((current_price - target) / volatility) * 100 if volatility > 0 else 0
                    reasons.append(f"변동성 돌파(+{breakout_ratio:.1f}%)")
            
            # 이동평균선 전략 분석
            if 'MA' in results:
                ma = results['MA']
                ma5 = ma.get('MA5', 0)
                ma20 = ma.get('MA20', 0)
                
                if ma5 > ma20:
                    ma_ratio = ((ma5 - ma20) / ma20) * 100
                    reasons.append(f"골든크로스(+{ma_ratio:.1f}%)")
                
                if current_price > ma5:
                    price_ratio = ((current_price - ma5) / ma5) * 100
                    reasons.append(f"단기 상승추세(+{price_ratio:.1f}%)")
            
            return reasons if reasons else None
            
        except Exception as e:
            logger.error(f"매수 이유 생성 중 오류: {str(e)}")
            return None

    def execute_buy(self, ticker, current_price, market_weight, confidence, buy_reasons=None):
        """매수 실행"""
        if ticker in self.held_coins:
            # 추가 매수 로직
            if not self.can_buy_more(ticker, current_price):
                return False
            return self._execute_additional_buy(ticker, current_price, market_weight)
        else:
            # 신규 매수 로직
            return self._execute_initial_buy(ticker, current_price, market_weight, confidence, buy_reasons)

    def _execute_initial_buy(self, ticker, current_price, market_weight, confidence, buy_reasons):
        """신규 매수 실행"""
        # 시장 상황에 따른 매수 금액 조정
        base_amount = self.settings.Trading.BASE_TRADE_AMOUNT
        if market_weight > 1.2:  # 강세장
            trade_amount = base_amount * 1.2
        elif market_weight < 0.8:  # 약세장
            trade_amount = base_amount * 0.8
        else:
            trade_amount = base_amount

        # 분할 매수 설정
        if confidence > 85:  # 매우 높은 신뢰도
            initial_ratio = 0.7  # 70% 매수
        elif confidence > 75:  # 높은 신뢰도
            initial_ratio = 0.5  # 50% 매수
        else:
            initial_ratio = 0.3  # 30% 매수

        # 실제 매수 금액 계산
        buy_amount = max(trade_amount * initial_ratio, self.settings.Trading.MIN_TRADE_AMOUNT)
        
        try:
            # 매수 주문
            order = self.upbit.buy_market_order(ticker, buy_amount)
            if order and 'error' not in order:
                # 매수 정보 저장
                self.held_coins[ticker] = {
                    'buy_price': current_price,
                    'buy_time': datetime.now(KST),
                    'buy_amount': buy_amount,
                    'buy_count': 1,
                    'investment': buy_amount,
                    'initial_confidence': confidence,
                    'remaining_ratio': 1 - initial_ratio,
                    'buy_reasons': buy_reasons
                }
                logger.info(f"{ticker} 신규 매수 성공: {buy_amount:,.0f}원 (신뢰도: {confidence:.1f}%)")
                return True
        except Exception as e:
            logger.error(f"{ticker} 매수 실패: {str(e)}")
        
        return False

    def _execute_additional_buy(self, ticker, current_price, market_weight):
        """추가 매수 실행"""
        coin_data = self.held_coins[ticker]
        initial_confidence = coin_data.get('initial_confidence', 0)
        remaining_ratio = coin_data.get('remaining_ratio', 0)
        
        if remaining_ratio <= 0:
            return False

        # 추가 매수 금액 계산
        base_amount = self.settings.Trading.BASE_TRADE_AMOUNT
        price_drop = (coin_data['buy_price'] - current_price) / coin_data['buy_price'] * 100
        
        # 가격 하락 정도에 따른 매수 비율 조정
        if price_drop > 2.0:
            buy_ratio = min(remaining_ratio, 0.5)  # 최대 50% 추가
        elif price_drop > 1.0:
            buy_ratio = min(remaining_ratio, 0.3)  # 최대 30% 추가
        else:
            buy_ratio = min(remaining_ratio, 0.2)  # 최대 20% 추가

        buy_amount = base_amount * buy_ratio
        
        try:
            # 추가 매수 주문
            order = self.upbit.buy_market_order(ticker, buy_amount)
            if order and 'error' not in order:
                # 매수 정보 업데이트
                coin_data['buy_count'] += 1
                coin_data['investment'] += buy_amount
                coin_data['remaining_ratio'] = remaining_ratio - buy_ratio
                
                # 평균단가 재계산
                total_investment = coin_data['investment']
                total_quantity = total_investment / current_price
                coin_data['buy_price'] = total_investment / total_quantity
                
                logger.info(f"{ticker} 추가 매수 성공: {buy_amount:,.0f}원 (하락률: {price_drop:.1f}%)")
                return True
        except Exception as e:
            logger.error(f"{ticker} 추가 매수 실패: {str(e)}")
        
        return False

    def should_trade_in_current_hour(self):
        """현재 시간대 거래 가능 여부 확인"""
        current_hour = datetime.now().hour
        
        # 거래 시간 확인
        if current_hour < Settings.Trading.TRADING_HOURS_START or current_hour >= Settings.Trading.TRADING_HOURS_END:
            return False
        
        # 점심시간 확인
        if Settings.Trading.AVOID_LUNCH_HOURS and 12 <= current_hour < 13:
            return False
        
        return True

    def can_buy_more(self, ticker, current_price):
        """추가 매수 가능 여부 확인"""
        if not self.settings.Trading.ENABLE_ADDITIONAL_BUY:
            return False

        coin_data = self.held_coins[ticker]
        
        # 최대 매수 횟수 확인
        if coin_data.get('buy_count', 1) >= self.settings.Trading.MAX_BUYS_PER_COIN:
            return False
            
        # 현재가가 이전 매수가 대비 설정된 비율 이상 하락했는지 확인
        price_drop = (coin_data['buy_price'] - current_price) / coin_data['buy_price'] * 100
        if price_drop < self.settings.Trading.ADDITIONAL_BUY_DIP:
            return False
            
        # 총 투자금액 확인
        current_investment = coin_data.get('investment', 0)
        if current_investment + self.settings.Trading.BASE_TRADE_AMOUNT > self.settings.Trading.MAX_POSITION_SIZE:
            return False
            
        return True

# 누적 수익률 계산 함수 추가
def update_cumulative_profit(buy_price, sell_price):
    """누적 수익률 계산 (수수료 고려)"""
    global cumulative_profit
    commission = Settings.Trading.COMMISSION_RATE
    profit_ratio = (sell_price * (1 - commission)) / (buy_price * (1 + commission))
    cumulative_profit *= profit_ratio
    return (profit_ratio - 1) * 100  # 수익률을 퍼센트로 반환

def get_ohlcv_cached(ticker, interval="minute5", count=100):
    """OHLCV 데이터 조회 (캐시 적용)"""
    cache_key = f"{ticker}_{interval}_{count}"
    cached = ohlcv_cache.get(cache_key)
    if cached is not None and not cached.empty:
        return cached

    df = safe_api_call(pyupbit.get_ohlcv, ticker, interval=interval, count=count)
    if df is not None and not df.empty:
        ohlcv_cache.set(cache_key, df)
        return df
    return None

def print_settings():
    """현재 설정값들을 로깅"""
    logger.info("====== 현재 설정값 ======")
    logger.info("[거래 설정]")
    logger.info(f"기본 거래금액: {Settings.Trading.BASE_TRADE_AMOUNT:,}원")
    logger.info(f"최소 거래금액: {Settings.Trading.MIN_TRADE_AMOUNT:,}원")
    logger.info(f"매수 신뢰도 기준: {Settings.Trading.CONFIDENCE_THRESHOLD}%")
    logger.info(f"익절 기준: {Settings.Trading.TAKE_PROFIT_PERCENT}%")
    logger.info(f"손절 기준: {Settings.Trading.STOP_LOSS_PERCENT}%")
    logger.info(f"강제 청산 시간: {Settings.Trading.FORCE_SELL_MINUTES}분")
    
    logger.info("\n[기술적 지표 설정]")
    logger.info(f"RSI 기간: {Settings.Technical.RSI_PERIOD}")
    logger.info(f"RSI 과매도 기준: {Settings.Technical.RSI_OVERSOLD}")
    logger.info(f"볼린저밴드 기간: {Settings.Technical.BB_WINDOW}")
    logger.info(f"볼린저밴드 표준편차: {Settings.Technical.BB_STD_DEV}")
    
    logger.info("\n[시장 설정]")
    logger.info(f"최소 거래량: {Settings.Market.MIN_TRADE_VOLUME:,}")
    logger.info(f"거래량 업데이트 주기: {Settings.Market.VOLUME_UPDATE_INTERVAL}분")
    logger.info(f"제외 코인: {', '.join(Settings.Market.STABLE_COINS)}")
    
    logger.info("\n[시스템 설정]")
    logger.info(f"API 요청 간격: {Settings.System.API_REQUEST_INTERVAL}초")
    logger.info(f"캐시 유효 기간: {Settings.System.CACHE_EXPIRY}초")
    logger.info(f"로그 파일 최대 크기: {Settings.System.LOG_ROTATION_SIZE/1024/1024:.1f}MB")
    logger.info(f"로그 파일 보관 개수: {Settings.System.LOG_BACKUP_COUNT}개")
    logger.info("========================\n")

class Indicators:
    """기술적 지표 계산 클래스"""
    
    @staticmethod
    def calculate_rsi(df, period=14):
        """RSI 계산"""
        try:
            closes = df['close'].values
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            if len(gains) < period:
                return None
            
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            if avg_loss == 0:
                return 100
                
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
            
        except Exception as e:
            logger.error(f"RSI 계산 중 오류: {str(e)}")
            return None
            
    @staticmethod
    def calculate_bollinger_bands(df, window=20, std_dev=2):
        """볼린저 밴드 계산"""
        try:
            rolling_mean = df['close'].rolling(window=window).mean()
            rolling_std = df['close'].rolling(window=window).std()
            
            if rolling_mean.isnull().all() or rolling_std.isnull().all():
                return None, None, None
                
            upper_band = rolling_mean + std_dev * rolling_std
            lower_band = rolling_mean - std_dev * rolling_std
            
            current_price = df['close'].iloc[-1]
            bb_range = upper_band.iloc[-1] - lower_band.iloc[-1]
            
            if bb_range == 0:
                bb_position = 50
            else:
                bb_position = ((current_price - lower_band.iloc[-1]) / bb_range) * 100
                
            return upper_band.iloc[-1], lower_band.iloc[-1], bb_position
            
        except Exception as e:
            logger.error(f"볼린저 밴드 계산 중 오류: {str(e)}")
            return None, None, None
            
    @staticmethod
    def calculate_volatility_breakout(df, k=0.5):
        """변동성 돌파 계산"""
        try:
            if len(df) < 2:
                return None, None
                
            yesterday = df.iloc[-2]
            today_open = df.iloc[-1]['open']
            
            # 전일 변동폭
            volatility = yesterday['high'] - yesterday['low']
            # 목표가 계산
            target_price = today_open + volatility * k
            
            return target_price, volatility
            
        except Exception as e:
            logger.error(f"변동성 돌파 계산 중 오류: {str(e)}")
            return None, None
            
    @staticmethod
    def calculate_moving_averages(df, periods=[5, 20]):
        """이동평균선 계산"""
        try:
            mas = {}
            for period in periods:
                ma = df['close'].rolling(window=period).mean()
                mas[f'MA{period}'] = ma.iloc[-1]
            return mas
            
        except Exception as e:
            logger.error(f"이동평균선 계산 중 오류: {str(e)}")
            return None

    @staticmethod
    def calculate_bb_width(df, window=20, std_dev=2):
        """볼린저 밴드 폭 계산"""
        try:
            rolling_mean = df['close'].rolling(window=window).mean()
            rolling_std = df['close'].rolling(window=window).std()
            
            upper_band = rolling_mean + std_dev * rolling_std
            lower_band = rolling_mean - std_dev * rolling_std
            
            # 밴드 폭을 중간값으로 나누어 정규화
            bb_width = (upper_band - lower_band) / rolling_mean
            
            return bb_width.iloc[-1]  # 최신 값만 반환
            
        except Exception as e:
            logger.error(f"볼린저 밴드 폭 계산 중 오류: {str(e)}")
            return None

class TradingStrategy:
    def __init__(self):
        self.volume_period = 20
        self.price_period = 10
        self.rsi_period = 14
        self.bb_period = 20
        self.bb_std = 2

    def analyze(self, ticker, df, current_price):
        try:
            if df is None or df.empty:
                return {'confidence': 0, 'signal': 'NONE'}
            
            result = self._calculate_indicators(df)
            if not result:
                return {'confidence': 0, 'signal': 'NONE'}
                
            confidence = self.calculate_confidence(result)
            signal = 'BUY' if confidence > 70 else 'NONE'
            
            return {**result, 'confidence': confidence, 'signal': signal}
            
        except Exception as e:
            logging.error(f"{ticker} 기본 분석 중 오류: {str(e)}")
            return {'confidence': 0, 'signal': 'NONE'}

    def _calculate_indicators(self, df):
        try:
            if df is None or df.empty:
                return None
                
            result = {}
            
            # RSI 계산
            rsi = self.calculate_rsi(df)
            if rsi is not None:
                result['rsi'] = rsi
                
            # 볼린저 밴드 계산
            bb_width = self.calculate_bb_width(df)
            if bb_width is not None:
                result['bb_width'] = bb_width
                
            # 거래량 비율 계산
            volume_ratio = self.calculate_volume_ratio(df)
            if volume_ratio is not None:
                result['volume_ratio'] = volume_ratio
                
            # 모멘텀 계산
            momentum = self.calculate_momentum(df)
            if momentum is not None:
                result['momentum'] = momentum
                
            return result if result else None
            
        except Exception as e:
            logging.error(f"지표 계산 중 오류: {str(e)}")
            return None

    def calculate_rsi(self, df):
        try:
            if df is None or df.empty or len(df) < self.rsi_period + 1:
                return None
                
            delta = df['close'].diff()
            if delta.empty:
                return None
                
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            
            if gain.empty or loss.empty or loss.iloc[-1] == 0:
                return None
                
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
            
        except Exception as e:
            logging.error(f"RSI 계산 중 오류: {str(e)}")
            return None

    def calculate_bb_width(self, df):
        try:
            if df is None or df.empty or len(df) < self.bb_period:
                return None
                
            ma = df['close'].rolling(window=self.bb_period).mean()
            std = df['close'].rolling(window=self.bb_period).std()
            
            if ma.empty or std.empty or pd.isna(ma.iloc[-1]) or pd.isna(std.iloc[-1]):
                return None
                
            upper = ma + self.bb_std * std
            lower = ma - self.bb_std * std
            
            width = (upper - lower) / ma
            return float(width.iloc[-1]) if not pd.isna(width.iloc[-1]) else None
            
        except Exception as e:
            logging.error(f"볼린저 밴드 폭 계산 중 오류: {str(e)}")
            return None

    def calculate_volume_ratio(self, df):
        try:
            if df is None or df.empty or len(df) < self.volume_period:
                return None
                
            avg_volume = df['volume'].rolling(window=self.volume_period).mean()
            if avg_volume.empty or pd.isna(avg_volume.iloc[-1]) or avg_volume.iloc[-1] == 0:
                return None
                
            current_volume = df['volume'].iloc[-1]
            ratio = current_volume / avg_volume.iloc[-1]
            
            return float(ratio) if not pd.isna(ratio) else None
            
        except Exception as e:
            logging.error(f"거래량 비율 계산 중 오류: {str(e)}")
            return None

    def calculate_momentum(self, df):
        try:
            if df is None or df.empty or len(df) < self.price_period + 1:
                return None
                
            momentum = df['close'].pct_change(periods=self.price_period)
            return float(momentum.iloc[-1]) if not pd.isna(momentum.iloc[-1]) else None
            
        except Exception as e:
            logging.error(f"모멘텀 계산 중 오류: {str(e)}")
            return None

    def calculate_confidence(self, data):
        """기본 신뢰도 계산 메서드 - 하위 클래스에서 재정의"""
        return 0

class RSIBBStrategy(TradingStrategy):
    def __init__(self, rsi_period=14, bb_period=20, bb_std=2):
        super().__init__()
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std

    def analyze(self, ticker, df, current_price):
        try:
            if df is None or df.empty:
                return {'confidence': 0, 'signal': 'NONE'}
                
            rsi = self.calculate_rsi(df)
            bb_width = self.calculate_bb_width(df)
            
            if rsi is None or bb_width is None:
                return {'confidence': 0, 'signal': 'NONE'}
                
            confidence = self.calculate_confidence({
                'rsi': rsi,
                'bb_width': bb_width
            })
            
            signal = 'BUY' if confidence > 70 else 'NONE'
            
            return {
                'confidence': confidence,
                'signal': signal,
                'rsi': rsi,
                'bb_width': bb_width
            }
        except Exception as e:
            logging.error(f"{ticker} RSI/BB 분석 중 오류: {str(e)}")
            return {'confidence': 0, 'signal': 'NONE'}

    def calculate_confidence(self, data):
        try:
            if not isinstance(data, dict) or 'rsi' not in data or 'bb_width' not in data:
                return 0
                
            rsi = data['rsi']
            bb_width = data['bb_width']
            
            if rsi is None or bb_width is None:
                return 0
                
            rsi_score = 100 - rsi if rsi > 50 else rsi  # RSI가 낮을수록 높은 점수
            bb_score = min(100, bb_width * 50)  # BB 밴드가 넓을수록 높은 점수
            
            # 가중 평균 계산
            confidence = (rsi_score * 0.6 + bb_score * 0.4)
            return max(0, min(100, confidence))
            
        except Exception as e:
            logging.error(f"신뢰도 계산 중 오류: {str(e)}")
            return 0

class VolumeBreakoutStrategy(TradingStrategy):
    def __init__(self, volume_period=20, price_period=10):
        super().__init__()
        self.volume_period = volume_period
        self.price_period = price_period

    def analyze(self, ticker, df, current_price):
        try:
            if df is None or df.empty:
                return {'confidence': 0, 'signal': 'NONE'}
                
            volume_ratio = self.calculate_volume_ratio(df)
            momentum = self.calculate_momentum(df)
            
            if volume_ratio is None or momentum is None:
                return {'confidence': 0, 'signal': 'NONE'}
                
            confidence = self.calculate_confidence({
                'volume_ratio': volume_ratio,
                'momentum': momentum
            })
            
            signal = 'BUY' if confidence > 70 else 'NONE'
            
            return {
                'confidence': confidence,
                'signal': signal,
                'volume_ratio': volume_ratio,
                'momentum': momentum
            }
        except Exception as e:
            logging.error(f"{ticker} 거래량 돌파 분석 중 오류: {str(e)}")
            return {'confidence': 0, 'signal': 'NONE'}

    def calculate_confidence(self, data):
        try:
            if not isinstance(data, dict) or 'volume_ratio' not in data or 'momentum' not in data:
                return 0
                
            volume_ratio = data['volume_ratio']
            momentum = data['momentum']
            
            if volume_ratio is None or momentum is None:
                return 0
                
            volume_score = min(100, volume_ratio * 50)  # 거래량이 높을수록 높은 점수
            momentum_score = min(100, momentum * 100)  # 모멘텀이 강할수록 높은 점수
            
            # 가중 평균 계산
            confidence = (volume_score * 0.6 + momentum_score * 0.4)
            return max(0, min(100, confidence))
            
        except Exception as e:
            logging.error(f"신뢰도 계산 중 오류: {str(e)}")
            return 0

class VolatilityBreakoutStrategy(TradingStrategy):
    """변동성 돌파 전략"""
    def analyze(self, ticker, df, current_price):
        try:
            target_price, volatility = Indicators.calculate_volatility_breakout(df, k=0.5)
            
            if target_price is None:
                return None
                
            return {
                'target_price': target_price,
                'volatility': volatility,
                'current_price': current_price,
                'strategy': 'VOLATILITY'
            }
            
        except Exception as e:
            logger.error(f"변동성 돌파 분석 중 오류: {str(e)}")
            return None
            
    def calculate_confidence(self, analysis_result):
        if not analysis_result:
            return 0
            
        target = analysis_result['target_price']
        current = analysis_result['current_price']
        volatility = analysis_result['volatility']
        
        # 횡보장에서는 보수적인 신뢰도 계산
        if current > target:
            if volatility == 0:
                return 40  # 기본값 하향 조정
                
            # 돌파 정도에 따른 신뢰도 계산 (보수적으로 조정)
            breakout_ratio = (current - target) / volatility
            if breakout_ratio <= 0.5:  # 약한 돌파
                confidence = 40 + breakout_ratio * 40  # 40-60점
            elif breakout_ratio <= 1.0:  # 중간 돌파
                confidence = 60 + (breakout_ratio - 0.5) * 30  # 60-75점
            else:  # 강한 돌파
                confidence = 75 + min(25, (breakout_ratio - 1.0) * 20)  # 75-100점
        else:
            confidence = 0
            
        return confidence

class MAStrategy(TradingStrategy):
    """이동평균선 전략"""
    def analyze(self, ticker, df, current_price):
        try:
            mas = Indicators.calculate_moving_averages(df, periods=[5, 20])
            
            if not mas:
                return None
                
            return {
                'MA5': mas['MA5'],
                'MA20': mas['MA20'],
                'current_price': current_price,
                'strategy': 'MA'
            }
            
        except Exception as e:
            logger.error(f"이동평균선 분석 중 오류: {str(e)}")
            return None
            
    def calculate_confidence(self, analysis_result):
        if not analysis_result:
            return 0
            
        ma5 = analysis_result['MA5']
        ma20 = analysis_result['MA20']
        current = analysis_result['current_price']
        
        # 골든 크로스 및 데드 크로스 확인
        if ma5 > ma20:  # 골든 크로스 상태
            # 현재가가 5일선 위에 있을 때 추가 신뢰도
            if current > ma5:
                confidence = 80
            else:
                confidence = 60
        else:  # 데드 크로스 상태
            confidence = 20
            
        return confidence

class StrategyComposer:
    def __init__(self):
        self.rsi_bb_strategy = RSIBBStrategy()
        self.volume_breakout_strategy = VolumeBreakoutStrategy()
        self.volatility_strategy = VolatilityBreakoutStrategy()
        self.ma_strategy = MAStrategy()

    def analyze(self, ticker, df, current_price):
        try:
            if df is None or df.empty:
                logging.error(f"{ticker} 데이터가 없거나 비어있습니다.")
                return {'confidence': 0, 'signal': 'NONE', 'reasons': []}

            market_condition = self._analyze_market_condition(df)
            if market_condition is None:
                logging.error(f"{ticker} 시장 상황 분석 실패")
                return {'confidence': 0, 'signal': 'NONE', 'reasons': []}

            strategy_weights = self._get_strategy_weights(market_condition)
            strategy_results = {}
            strategy_reasons = []
            total_weight = 0
            total_confidence = 0

            # RSI/BB 전략 분석
            rsi_bb_result = self.rsi_bb_strategy.analyze(ticker, df, current_price)
            if rsi_bb_result and 'confidence' in rsi_bb_result:
                weight = strategy_weights.get('rsi_bb', 0)
                total_confidence += rsi_bb_result['confidence'] * weight
                total_weight += weight
                strategy_results['rsi_bb'] = rsi_bb_result
                if rsi_bb_result['confidence'] > 0:
                    strategy_reasons.append(self._get_strategy_reason('RSI/BB', rsi_bb_result, market_condition))

            # 거래량 돌파 전략 분석
            volume_result = self.volume_breakout_strategy.analyze(ticker, df, current_price)
            if volume_result and 'confidence' in volume_result:
                weight = strategy_weights.get('volume_breakout', 0)
                total_confidence += volume_result['confidence'] * weight
                total_weight += weight
                strategy_results['volume_breakout'] = volume_result
                if volume_result['confidence'] > 0:
                    strategy_reasons.append(self._get_strategy_reason('Volume Breakout', volume_result, market_condition))

            # 변동성 돌파 전략 분석
            volatility_result = self.volatility_strategy.analyze(ticker, df, current_price)
            if volatility_result and 'confidence' in volatility_result:
                weight = strategy_weights.get('volatility', 0)
                total_confidence += volatility_result['confidence'] * weight
                total_weight += weight
                strategy_results['volatility'] = volatility_result
                if volatility_result['confidence'] > 0:
                    strategy_reasons.append(self._get_strategy_reason('Volatility', volatility_result, market_condition))

            # 이동평균선 전략 분석
            ma_result = self.ma_strategy.analyze(ticker, df, current_price)
            if ma_result and 'confidence' in ma_result:
                weight = strategy_weights.get('ma', 0)
                total_confidence += ma_result['confidence'] * weight
                total_weight += weight
                strategy_results['ma'] = ma_result
                if ma_result['confidence'] > 0:
                    strategy_reasons.append(self._get_strategy_reason('MA', ma_result, market_condition))

            # 최종 신뢰도 계산
            if total_weight > 0:
                final_confidence = total_confidence / total_weight
            else:
                final_confidence = 0

            self._log_strategy_analysis(ticker, market_condition, strategy_reasons, final_confidence)

            return {
                'confidence': final_confidence,
                'signal': 'BUY' if final_confidence > Settings.Trading.CONFIDENCE_THRESHOLD else 'NONE',
                'reasons': strategy_reasons,
                'market_condition': market_condition
            }

        except Exception as e:
            logging.error(f"{ticker} 전략 분석 중 오류: {str(e)}")
            return {'confidence': 0, 'signal': 'NONE', 'reasons': []}

    def _analyze_market_condition(self, df):
        """
        시장 상황을 분석하여 현재 시장 상태를 반환합니다.
        """
        try:
            # 기본값 설정
            market_condition = MarketCondition.NORMAL
            
            # 데이터가 충분한지 확인
            if len(df) < 20:
                logging.warning("데이터가 부족하여 기본 시장 상황을 반환합니다.")
                return market_condition
            
            # 추세 분석
            ma_short = df['close'].rolling(window=5).mean()
            ma_long = df['close'].rolling(window=20).mean()
            
            # 변동성 분석
            volatility = df['close'].pct_change().std() * np.sqrt(252)
            
            # 거래량 분석
            volume_ma = df['volume'].rolling(window=20).mean()
            current_volume = df['volume'].iloc[-1]
            
            # 시장 상태 결정
            if volatility > 0.5:  # 높은 변동성
                market_condition = MarketCondition.VOLATILE
            elif ma_short.iloc[-1] > ma_long.iloc[-1] * 1.02:  # 상승 추세
                market_condition = MarketCondition.TRENDING
            elif current_volume > volume_ma.mean() * 2:  # 거래량 급증
                market_condition = MarketCondition.VOLATILE
            else:
                market_condition = MarketCondition.NORMAL
            
            return market_condition
            
        except Exception as e:
            logging.error(f"시장 상황 분석 중 오류 발생: {str(e)}")
            return MarketCondition.NORMAL

    def _get_strategy_weights(self, market_condition):
        """
        시장 상황에 따른 전략별 가중치를 반환합니다.
        """
        try:
            weights = {
                MarketCondition.NORMAL: {
                    'RSI': 0.3,
                    'BB': 0.3,
                    'VOLUME': 0.2,
                    'MA': 0.2
                },
                MarketCondition.UPTREND: {
                    'RSI': 0.2,
                    'BB': 0.2,
                    'VOLUME': 0.2,
                    'MA': 0.4
                },
                MarketCondition.DOWNTREND: {
                    'RSI': 0.3,
                    'BB': 0.3,
                    'VOLUME': 0.2,
                    'MA': 0.2
                }
            }
            return weights.get(market_condition, weights[MarketCondition.NORMAL])
        except Exception as e:
            logging.error(f"시장 상황별 가중치 계산 중 오류 발생: {e}")
            return weights[MarketCondition.NORMAL]  # 기본값 반환

    def _get_strategy_reason(self, strategy_name, result, market_condition):
        """각 전략별 선택 근거 생성"""
        if strategy_name == 'rsi_bb':
            rsi = result.get('rsi', 0)
            bb_pos = result.get('bb_position', 50)
            if market_condition['trend'] == 'DOWNWARD' and rsi < 30:
                return f"과매도 상태(RSI: {rsi:.1f})에서 반등 기대"
            elif market_condition['trend'] == 'UPWARD' and bb_pos > 80:
                return f"과매수 상태(BB: {bb_pos:.1f}%)로 신중한 접근 필요"
            return f"RSI({rsi:.1f})와 BB 위치({bb_pos:.1f}%)가 {market_condition['trend']} 추세에 적합"
            
        elif strategy_name == 'volume_breakout':
            volume_ratio = result.get('volume_ratio', 0)
            momentum = result.get('momentum', 0)
            if market_condition['volume'] == 'HIGH':
                return f"거래량 급등({volume_ratio:.1f}배), 상승 모멘텀({momentum*100:.1f}%)"
            return f"거래량 증가({volume_ratio:.1f}배) 감지"
            
        elif strategy_name == 'volatility':
            target = result.get('target_price', 0)
            volatility = result.get('volatility', 0)
            if market_condition['volatility'] == 'HIGH':
                return f"변동성 높음({volatility:.1f}) 전략 유효"
            return f"변동성({volatility:.1f}) 기반 목표가 돌파 시 매수 신호"
            
        elif strategy_name == 'ma':
            ma5 = result.get('MA5', 0)
            ma20 = result.get('MA20', 0)
            if ma5 > ma20:
                return f"골든크로스 상태에서 추세 추종"
            return f"이동평균선 배열이 {market_condition['trend']} 추세와 일치"
            
        return ""

    def _log_strategy_analysis(self, ticker, market_condition, strategy_reasons, total_confidence):
        """전략 분석 결과 로깅"""
        # 매수 신호 발생 여부 확인 (신뢰도가 기준값 이상인 경우)
        is_buy_signal = total_confidence >= Settings.Trading.CONFIDENCE_THRESHOLD

        if is_buy_signal:
            # 매수 신호가 발생한 경우에만 상세 로그 출력
            log_msg = [
                f"\n{ticker} 매수 신호 감지!",
                f"시장 상황: {market_condition}",
                f"종합 신뢰도: {total_confidence:.1f}%"
            ]
            
            if strategy_reasons:
                log_msg.append("\n채택된 전략:")
                for sr in strategy_reasons:
                    log_msg.append(
                        f"- {sr['strategy']}: 가중치({sr['weight']:.1f}) × "
                        f"신뢰도({sr['confidence']:.1f}%) = {sr['weight'] * sr['confidence']:.1f}%"
                    )
                    log_msg.append(f"  → {sr['reason']}")
            
            logger.info('\n'.join(log_msg))
        else:
            # 매수 신호가 없는 경우 간단한 로그만 출력
            logger.debug(f"{ticker} 분석 중... (신뢰도: {total_confidence:.1f}%)")

class DataManager:
    def __init__(self, settings):
        self.settings = settings
        self.base_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 디렉토리
        self.log_file = self.setup_data_logging()
        self.last_stats_cycle = 0
        
    def setup_data_logging(self):
        """데이터 로깅 설정"""
        # 헤더 정의를 try 블록 밖으로 이동
        headers = [
            "timestamp", "cycle_num", 
            "ticker", "action", 
            "price", "amount",
            "total_value", "buy_price",
            "profit_loss_percent", "profit_loss_amount",
            "holding_time", "sell_reason",
            "market_condition", "atr_value"
        ]
        
        try:
            # 로그 파일 경로 설정
            current_time = datetime.now(KST).strftime("%Y%m%d_%H%M")
            log_filename = f"trading_data_{current_time}.csv"
            log_file_path = os.path.join(self.base_path, log_filename)
            
            # 파일 생성
            with open(log_file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            
            logger.info(f"거래 데이터 로그 파일 생성 완료: {log_file_path}")
            return log_file_path
            
        except Exception as e:
            logger.error(f"로그 파일 생성 중 오류 발생: {str(e)}")
            # 오류 발생 시 현재 디렉토리에 시도
            fallback_path = f"trading_data_{current_time}.csv"
            logger.info(f"대체 경로에 파일 생성 시도: {fallback_path}")
            
            with open(fallback_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            
            return fallback_path
            
    def log_trade_data(self, data_dict):
        """거래 데이터 CSV 저장"""
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                row = [
                    data_dict.get(field, '') for field in [
                        "timestamp", "cycle_num", 
                        "ticker", "action", 
                        "price", "amount",
                        "total_value", "buy_price",
                        "profit_loss_percent", "profit_loss_amount",
                        "holding_time", "sell_reason",
                        "market_condition", "atr_value"
                    ]
                ]
                writer.writerow(row)
                logger.debug(f"거래 데이터 기록 완료: {data_dict['ticker']} - {data_dict['action']}")
        except Exception as e:
            logger.error(f"데이터 로깅 중 오류 발생: {str(e)}")
            
    def calculate_statistics(self, current_cycle):
        """주기적 통계 계산 (50 사이클마다)"""  # 100에서 50으로 변경
        if current_cycle % 50 != 0 or current_cycle == self.last_stats_cycle:
            return
            
        try:
            df = pd.read_csv(self.log_file)
            cycle_start = max(0, current_cycle - 50)
            cycle_data = df[(df['cycle_num'] >= cycle_start) & (df['cycle_num'] <= current_cycle)]
            
            if len(cycle_data) == 0:
                return
                
            # 기본 통계
            total_trades = len(cycle_data)
            winning_trades = len(cycle_data[cycle_data['profit_loss_percent'] > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # 수익률 통계
            avg_profit = cycle_data['profit_loss_percent'].mean()
            max_profit = cycle_data['profit_loss_percent'].max()
            max_loss = cycle_data['profit_loss_percent'].min()
            profit_std = cycle_data['profit_loss_percent'].std()  # 수익률 표준편차
            
            # 수익금 통계
            total_profit = cycle_data['profit_loss_amount'].sum()
            avg_profit_amount = cycle_data['profit_loss_amount'].mean()
            
            # 보유시간 통계
            avg_hold_time = cycle_data['holding_time'].mean()
            max_hold_time = cycle_data['holding_time'].max()
            
            # 매도 사유 분석
            sell_reasons = cycle_data['sell_reason'].value_counts()
            
            # 결과 출력
            logger.info("\n" + "="*50)
            logger.info(f"거래 통계 요약 (사이클 {cycle_start} - {current_cycle})")
            logger.info("="*50)
            
            logger.info("\n[기본 통계]")
            logger.info(f"총 거래 수: {total_trades}건")
            logger.info(f"승률: {win_rate:.2f}% ({winning_trades}/{total_trades})")
            logger.info(f"평균 수익률: {avg_profit:.2f}%")
            logger.info(f"수익률 표준편차: {profit_std:.2f}%")
            
            logger.info("\n[수익률 상세]")
            logger.info(f"최고 수익률: {max_profit:.2f}%")
            logger.info(f"최대 손실률: {max_loss:.2f}%")
            logger.info(f"누적 순수익: {total_profit:,.0f}원")
            logger.info(f"평균 수익금: {avg_profit_amount:,.0f}원")
            
            logger.info("\n[보유시간]")
            logger.info(f"평균 보유시간: {avg_hold_time:.1f}분")
            logger.info(f"최장 보유시간: {max_hold_time:.1f}분")
            
            logger.info("\n[매도 사유 분석]")
            for reason, count in sell_reasons.items():
                logger.info(f"- {reason}: {count}건")
            
            logger.info("\n" + "="*50 + "\n")
            
            self.last_stats_cycle = current_cycle
            
        except Exception as e:
            logger.error(f"통계 계산 중 오류: {str(e)}")
            
    def monitor_performance(self, current_cycle):
        """실시간 성능 모니터링 (5 사이클마다)"""
        if current_cycle % 5 != 0:  # 10에서 5로 변경
            return
            
        try:
            current_time = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
            logger.info("\n" + "-"*50)
            logger.info(f"실시간 모니터링 ({current_time}, 사이클: {current_cycle})")
            logger.info("-"*50)
            
            # 현재 보유 중인 코인 상태 출력
            if not trading_logic.held_coins:
                logger.info("보유 중인 코인 없음")
            else:
                total_profit = 0
                for ticker, data in trading_logic.held_coins.items():
                    current_price = get_current_price_safely(ticker)
                    if current_price is None:
                        continue
                        
                    buy_price = data['buy_price']
                    amount = data['amount']
                    buy_time = data['buy_time']
                    holding_time = (datetime.now(KST) - buy_time).total_seconds() / 60
                    current_profit = ((current_price - buy_price) / buy_price) * 100
                    profit_amount = (current_price - buy_price) * amount
                    total_profit += profit_amount
                    
                    logger.info(f"\n{ticker}:")
                    logger.info(f"- 현재가: {current_price:,.2f}원 (매수가: {buy_price:,.2f}원)")
                    logger.info(f"- 수익률: {current_profit:+.2f}% ({profit_amount:+,.0f}원)")
                    logger.info(f"- 보유수량: {amount:.8f}")
                    logger.info(f"- 보유시간: {holding_time:.1f}분")
                    if 'buy_count' in data:
                        logger.info(f"- 매수횟수: {data['buy_count']}회")
                
                logger.info(f"\n현재 총 평가손익: {total_profit:+,.0f}원")
            
            # KRW 잔고 확인
            krw_balance = safe_api_call(upbit.get_balance, "KRW")
            if krw_balance is not None:
                logger.info(f"KRW 잔고: {krw_balance:,.0f}원")
            
            logger.info("-"*50 + "\n")
            
        except Exception as e:
            logger.error(f"성능 모니터링 중 오류: {str(e)}")

    def manage_data_files(self):
        """오래된 데이터 파일 관리 (매일)"""
        try:
            old_files = glob.glob("trading_data_*.csv")
            current_time = datetime.now()
            
            for file in old_files:
                try:
                    file_time = datetime.strptime(file.split('_')[2].split('.')[0], "%Y%m%d")
                    if (current_time - file_time).days > 30:
                        zip_name = f"{file[:-4]}.zip"
                        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                            zipf.write(file)
                        os.remove(file)
                        logger.info(f"오래된 데이터 파일 압축: {file} -> {zip_name}")
                except Exception as e:
                    logger.error(f"파일 처리 중 오류 ({file}): {str(e)}")
                    
        except Exception as e:
            logger.error(f"데이터 파일 관리 중 오류: {str(e)}")

    def analyze_trading_performance(self):
        """거래 데이터 분석 및 전략 최적화 제안"""
        try:
            df = pd.read_csv(self.log_file)
            if len(df) < 10:  # 최소 10개 이상의 거래 데이터 필요
                return
            
            # 1. 기본 통계
            total_trades = len(df)
            profitable_trades = len(df[df['profit_loss_percent'] > 0])
            win_rate = (profitable_trades / total_trades) * 100
            
            # 2. 수익률 분석
            avg_profit = df[df['profit_loss_percent'] > 0]['profit_loss_percent'].mean()
            avg_loss = df[df['profit_loss_percent'] < 0]['profit_loss_percent'].mean()
            max_profit = df['profit_loss_percent'].max()
            max_loss = df['profit_loss_percent'].min()
            
            # 3. 매도 사유 분석
            sell_reasons = df['sell_reason'].value_counts()
            
            # 4. 시장 상황별 분석
            market_conditions = df['market_condition'].value_counts()
            market_profit = df.groupby('market_condition')['profit_loss_percent'].mean()
            
            # 5. 보유 시간 분석
            avg_holding_time = df['holding_time'].mean()
            profit_holding_time = df[df['profit_loss_percent'] > 0]['holding_time'].mean()
            
            # 분석 결과 로깅
            logger.info("\n=== 거래 데이터 분석 결과 ===")
            logger.info(f"총 거래 수: {total_trades}건")
            logger.info(f"승률: {win_rate:.1f}%")
            logger.info(f"평균 수익: +{avg_profit:.2f}% / 평균 손실: {avg_loss:.2f}%")
            logger.info(f"최대 수익: +{max_profit:.2f}% / 최대 손실: {max_loss:.2f}%")
            logger.info(f"\n매도 사유 분포:")
            for reason, count in sell_reasons.items():
                logger.info(f"- {reason}: {count}건 ({count/total_trades*100:.1f}%)")
            
            logger.info(f"\n시장 상황별 수익률:")
            for condition in market_profit.index:
                logger.info(f"- {condition}: {market_profit[condition]:.2f}%")
            
            logger.info(f"\n보유 시간 분석:")
            logger.info(f"평균 보유 시간: {avg_holding_time:.1f}분")
            logger.info(f"수익 거래 평균 보유 시간: {profit_holding_time:.1f}분")
            
            # 전략 최적화 제안
            suggestions = []
            
            if win_rate < 40:
                suggestions.append("매수 신뢰도 기준을 상향 조정하세요.")
            
            if len(sell_reasons.get('FORCE_SELL', 0)) / total_trades > 0.3:
                suggestions.append("강제 청산 시간을 단축하는 것이 좋습니다.")
            
            if len(sell_reasons.get('DYNAMIC_SL', 0)) / total_trades > 0.2:
                suggestions.append("동적 손절 기준을 완화하세요.")
            
            if avg_holding_time < 10 and win_rate < 45:
                suggestions.append("최소 보유 시간을 증가시키세요.")
            
            if market_profit.get('STRONG_UP', 0) > 1.0:
                suggestions.append("상승장에서 더 공격적인 매수를 고려하세요.")
            
            if suggestions:
                logger.info("\n=== 전략 개선 제안 ===")
                for i, suggestion in enumerate(suggestions, 1):
                    logger.info(f"{i}. {suggestion}")
            
        except Exception as e:
            logger.error(f"거래 데이터 분석 중 오류: {str(e)}")

    def optimize_strategy(self):
        """전략 자동 최적화"""
        try:
            df = pd.read_csv(self.log_file)
            if len(df) < 20:  # 최소 20개 이상의 거래 데이터 필요
                return
            
            # 수익성이 높은 거래의 특징 분석
            profitable_trades = df[df['profit_loss_percent'] > 1.0]
            if len(profitable_trades) > 0:
                # 평균 보유 시간 계산
                optimal_holding_time = profitable_trades['holding_time'].mean()
                # 주요 매도 사유 분석
                best_sell_reasons = profitable_trades['sell_reason'].value_counts()
                # 성공적인 시장 상황 분석
                best_market_conditions = profitable_trades['market_condition'].value_counts()
                
                logger.info("\n=== 최적 전략 분석 ===")
                logger.info(f"최적 보유 시간: {optimal_holding_time:.1f}분")
                logger.info("성공적인 매도 사유:")
                for reason, count in best_sell_reasons.items():
                    logger.info(f"- {reason}: {count}건")
                logger.info("성공적인 시장 상황:")
                for condition, count in best_market_conditions.items():
                    logger.info(f"- {condition}: {count}건")
            
        except Exception as e:
            logger.error(f"전략 최적화 중 오류: {str(e)}")

class BacktestingLogic:
    def __init__(self, settings):
        self.settings = settings
        self.strategy_composer = StrategyComposer()
        self.trades = []
        self.current_position = None
        self.initial_balance = 1000000  # 초기 자본금
        self.current_balance = self.initial_balance
        
    def run_backtest(self, ticker, start_date, end_date):
        try:
            logging.info(f"\n==================================================")
            logging.info(f"{ticker} 백테스팅 시작")
            logging.info(f"기간: {start_date} ~ {end_date}")
            logging.info(f"초기자본: {self.initial_balance:,}원")
            logging.info(f"==================================================")
            
            # 데이터 로드
            df = self._load_data(ticker, start_date, end_date)
            if df is None or df.empty:
                logging.error("데이터 로드 실패")
                return None
                
            # 백테스팅 실행
            for i in range(len(df)):
                current_time = df.index[i]
                current_price = df['close'].iloc[i]
                
                # 현재 포지션이 있는 경우 매도 조건 확인
                if self.current_position:
                    self._check_sell_conditions(ticker, current_time, current_price)
                
                # 매수 조건 확인
                if not self.current_position:
                    self._check_buy_conditions(ticker, df.iloc[:i+1], current_time, current_price)
            
            # 최종 결과 분석 및 출력
            results = self._analyze_results()
            self._print_results(ticker)
            
            return results
            
        except Exception as e:
            logging.error(f"백테스팅 중 오류 발생: {str(e)}")
            return None
            
    def _load_data(self, ticker, start_date, end_date):
        try:
            # 데이터 로드 (1시간 봉 기준)
            df = pyupbit.get_ohlcv(ticker, interval="minute60", to=end_date, count=500)
            if df is None or df.empty:
                return None
                
            # 데이터 전처리
            df = df.loc[start_date:end_date]
            if df.empty:
                return None
                
            return df
            
        except Exception as e:
            logging.error(f"데이터 로드 중 오류: {str(e)}")
            return None
            
    def _check_buy_conditions(self, ticker, df, current_time, current_price):
        try:
            # 전략 분석
            analysis = self.strategy_composer.analyze(ticker, df, current_price)
            if not analysis:
                return
                
            # 매수 신호 확인
            if analysis['signal'] == 'BUY' and analysis['confidence'] > self.settings.Trading.CONFIDENCE_THRESHOLD:
                self._execute_buy(ticker, current_time, current_price)
                
        except Exception as e:
            logging.error(f"매수 조건 확인 중 오류: {str(e)}")
            
    def _check_sell_conditions(self, ticker, current_time, current_price):
        try:
            if not self.current_position:
                return
                
            buy_price = self.current_position['buy_price']
            buy_time = self.current_position['buy_time']
            
            # 수익률 계산
            profit_percent = (current_price - buy_price) / buy_price * 100
            
            # 익절
            if profit_percent >= self.settings.Trading.TAKE_PROFIT_PERCENT:
                self._execute_sell(ticker, current_time, current_price, "TAKE_PROFIT")
                return
                
            # 손절
            if profit_percent <= -self.settings.Trading.STOP_LOSS_PERCENT:
                self._execute_sell(ticker, current_time, current_price, "STOP_LOSS")
                return
                
            # 강제 청산
            holding_minutes = (current_time - buy_time).total_seconds() / 60
            if holding_minutes >= self.settings.Trading.FORCE_SELL_MINUTES:
                self._execute_sell(ticker, current_time, current_price, "FORCE_SELL")
                
        except Exception as e:
            logging.error(f"매도 조건 확인 중 오류: {str(e)}")
            
    def _execute_buy(self, ticker, time, price):
        try:
            if self.current_balance < self.settings.Trading.BASE_TRADE_AMOUNT:
                return
                
            amount = min(self.settings.Trading.BASE_TRADE_AMOUNT, self.current_balance)
            quantity = amount / price
            
            self.current_position = {
                'ticker': ticker,
                'buy_time': time,
                'buy_price': price,
                'quantity': quantity,
                'amount': amount
            }
            
            self.current_balance -= amount
            
        except Exception as e:
            logging.error(f"매수 실행 중 오류: {str(e)}")
            
    def _execute_sell(self, ticker, time, price, reason):
        try:
            if not self.current_position:
                return
                
            amount = self.current_position['quantity'] * price
            profit = amount - self.current_position['amount']
            profit_percent = profit / self.current_position['amount'] * 100
            
            self.trades.append({
                'ticker': ticker,
                'buy_time': self.current_position['buy_time'],
                'buy_price': self.current_position['buy_price'],
                'sell_time': time,
                'sell_price': price,
                'profit': profit,
                'profit_percent': profit_percent,
                'reason': reason
            })
            
            self.current_balance += amount
            self.current_position = None
            
        except Exception as e:
            logging.error(f"매도 실행 중 오류: {str(e)}")
            
    def _analyze_results(self):
        try:
            if not self.trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'total_return': 0,
                    'avg_profit': 0,
                    'avg_loss': 0,
                    'profit_factor': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0
                }
                
            # 기본 통계
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t['profit'] > 0])
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
            
            # 수익성 분석
            profits = [t['profit'] for t in self.trades if t['profit'] > 0]
            losses = [t['profit'] for t in self.trades if t['profit'] <= 0]
            
            avg_profit = sum(profits) / len(profits) if profits else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            profit_factor = abs(sum(profits) / sum(losses)) if losses and sum(losses) != 0 else 0
            
            # 리스크 분석
            returns = [t['profit_percent'] for t in self.trades]
            cumulative_returns = np.cumsum(returns)
            max_drawdown = min(0, min(cumulative_returns - np.maximum.accumulate(cumulative_returns)))
            
            # 샤프 비율
            if len(returns) > 1:
                returns_std = np.std(returns)
                sharpe_ratio = np.mean(returns) / returns_std * np.sqrt(252) if returns_std != 0 else 0
            else:
                sharpe_ratio = 0
                
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_return': (self.current_balance - self.initial_balance) / self.initial_balance * 100,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio
            }
            
        except Exception as e:
            logging.error(f"결과 분석 중 오류: {str(e)}")
            return None
            
    def _print_results(self, ticker):
        try:
            results = self._analyze_results()
            if not results:
                return
                
            logging.info(f"\n==================================================")
            logging.info(f"{ticker} 백테스팅 결과")
            logging.info(f"==================================================")
            
            logging.info(f"\n[기본 통계]")
            logging.info(f"총 거래 수: {results['total_trades']}건")
            logging.info(f"승률: {results['win_rate']:.1f}%")
            logging.info(f"총 수익률: {results['total_return']:.2f}%")
            
            logging.info(f"\n[수익성 분석]")
            logging.info(f"평균 수익: +{results['avg_profit']:.2f}%")
            logging.info(f"평균 손실: {results['avg_loss']:.2f}%")
            logging.info(f"수익 팩터: {results['profit_factor']:.2f}")
            
            logging.info(f"\n[리스크 분석]")
            logging.info(f"최대 낙폭: {results['max_drawdown']:.2f}%")
            logging.info(f"샤프 비율: {results['sharpe_ratio']:.2f}")
            
            logging.info(f"\n==================================================")
            
        except Exception as e:
            logging.error(f"결과 출력 중 오류: {str(e)}")

def run_backtest_analysis(start_date, end_date, tickers=None):
    """백테스트 분석 실행"""
    try:
        # 초기화
        total_trades = 0
        total_win_rate = 0
        total_return = 0
        total_sharpe = 0
        best_coin = {'ticker': None, 'return': -float('inf')}
        worst_coin = {'ticker': None, 'return': float('inf')}
        
        # 전체 마켓 정보 로드
        markets = load_markets()
        if not markets:
            logging.error("마켓 정보 로드 실패")
            return
            
        # 거래량 기준 필터링
        filtered_markets = filter_by_volume(markets)
        if not filtered_markets:
            logging.error("거래량 기준을 만족하는 코인이 없습니다")
            return
            
        logging.info(f"전체 KRW 마켓 조회, 대상 코인 수: {len(markets)}")
        logging.info(f"거래량 기준 필터링 후 대상 코인 수: {len(filtered_markets)}")
        
        # 백테스트 실행
        settings = Settings()
        successful_tests = 0
        
        for ticker in (tickers or filtered_markets):
            try:
                backtest = BacktestingLogic(settings)
                results = backtest.run_backtest(ticker, start_date, end_date)
                
                if results and results['total_trades'] > 0:
                    total_trades += results['total_trades']
                    total_win_rate += results['win_rate']
                    total_return += results['total_return']
                    total_sharpe += results['sharpe_ratio']
                    successful_tests += 1
                    
                    # 최고/최저 성과 코인 업데이트
                    if results['total_return'] > best_coin['return']:
                        best_coin = {'ticker': ticker, 'return': results['total_return']}
                    if results['total_return'] < worst_coin['return']:
                        worst_coin = {'ticker': ticker, 'return': results['total_return']}
                        
            except Exception as e:
                logging.error(f"{ticker} 백테스트 중 오류: {str(e)}")
                continue
                
        # 종합 결과 출력
        if successful_tests > 0:
            logging.info("\n==================================================")
            logging.info("백테스트 종합 결과")
            logging.info("==================================================")
            
            logging.info(f"\n[전체 통계]")
            logging.info(f"분석 완료된 코인 수: {successful_tests}개")
            logging.info(f"총 거래 수: {total_trades}건")
            logging.info(f"평균 승률: {total_win_rate/successful_tests:.1f}%")
            logging.info(f"평균 수익률: {total_return/successful_tests:.2f}%")
            logging.info(f"평균 샤프 비율: {total_sharpe/successful_tests:.2f}")
            
            logging.info(f"\n[코인별 성과]")
            logging.info(f"최고 성과 코인: {best_coin['ticker']} ({best_coin['return']:.2f}%)")
            logging.info(f"최저 성과 코인: {worst_coin['ticker']} ({worst_coin['return']:.2f}%)")
            
            logging.info("\n==================================================")
            
    except Exception as e:
        logging.error(f"백테스트 분석 중 오류 발생: {str(e)}")

# 전역 변수: Upbit 클라이언트
upbit = None
trading_logic = None
data_manager = None
emergency_stop = False

def main():
    """메인 함수"""
    try:
        global upbit, trading_logic, data_manager
        
        # 로깅 설정
        logger = setup_logging()
        
        # 환경 변수 로드
        load_dotenv()
        
        # API 키 유효성 검사
        if not UPBIT_ACCESS or not UPBIT_SECRET:
            raise ValueError("API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
            
        # Upbit 객체 초기화
        try:
            upbit = pyupbit.Upbit(UPBIT_ACCESS, UPBIT_SECRET)
            balance = safe_api_call(upbit.get_balance, "KRW")
            if balance is None:
                raise ValueError("잔고 조회 실패. API 키를 확인해주세요.")
            logger.info(f"초기 KRW 잔고: {balance:,.0f}원")
        except Exception as e:
            raise ValueError(f"Upbit 초기화 실패: {str(e)}")
        
        # 설정 초기화
        settings = Settings()
        
        # 설정값 출력 (한 번만)
        print_settings()
        
        # 데이터 매니저와 트레이딩 로직 인스턴스 생성
        data_manager = DataManager(settings)
        trading_logic = TradingLogic(settings, data_manager)
        
        logger.info("트레이딩 봇이 시작되었습니다.")
        
        # API 요청 카운터 및 시간 추적
        request_count = 0
        last_request_time = time.time()
        error_count = 0  # 연속 오류 카운터
        cycle_count = 0  # 사이클 카운터 추가
        
        while True:
            try:
                cycle_count += 1
                current_time = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"\n{'='*30}")
                logger.info(f"사이클 {cycle_count} 시작 ({current_time})")
                logger.info(f"{'='*30}")
                
                current_time = time.time()
                
                # API 요청 간격 조절
                if current_time - last_request_time < Settings.System.API_REQUEST_INTERVAL:
                    time.sleep(Settings.System.API_REQUEST_INTERVAL - (current_time - last_request_time))
                
                # 현재 보유 코인 수 확인
                current_holdings = len(trading_logic.held_coins)
                logger.info(f"현재 보유 코인 수: {current_holdings}/{Settings.Trading.MAX_HOLDINGS}")
                
                # 보유 중인 코인에 대해 매도 조건 확인
                if current_holdings > 0:
                    logger.info("\n[매도 조건 검사 시작]")
                    for ticker in list(trading_logic.held_coins.keys()):
                        logger.info(f"{ticker} 매도 조건 검사 중...")
                        current_price = get_current_price_safely(ticker)
                        if current_price is None:
                            error_count += 1
                            if error_count >= 5:
                                logger.warning("연속 API 오류 발생, 잠시 대기합니다.")
                                time.sleep(Settings.System.API_ERROR_INTERVAL * 2)
                                error_count = 0
                            continue
                        
                        error_count = 0  # 성공 시 오류 카운터 리셋
                        should_sell_result, reason = trading_logic.should_sell(ticker, current_price)
                        if should_sell_result:
                            logger.info(f"{ticker} 매도 시도 중... (사유: {reason})")
                            if trading_logic.execute_sell(ticker, current_price, reason):
                                time.sleep(Settings.System.API_REQUEST_INTERVAL)
                                current_holdings -= 1
                
                # 매수 가능한 코인 탐색 및 매수
                if current_holdings < Settings.Trading.MAX_HOLDINGS:
                    available_slots = Settings.Trading.MAX_HOLDINGS - current_holdings
                    logger.info(f"\n[매수 조건 검사 시작] (남은 슬롯: {available_slots}개)")
                    
                    # 전체 마켓 조회
                    all_tickers = pyupbit.get_tickers(fiat="KRW")
                    # 스테이블 코인 제외
                    tickers = [ticker for ticker in all_tickers if ticker not in Settings.Market.STABLE_COINS]
                    # 거래량 기준 필터링
                    tickers = filter_by_volume(tickers)
                    
                    buy_candidates = []
                    analyzed_count = 0
                    
                    # 매수 후보 선정 (병렬 처리 또는 일부만 분석)
                    for ticker in tickers[:30]:  # 상위 30개 코인만 분석
                        if ticker not in trading_logic.held_coins:
                            current_price = get_current_price_safely(ticker)
                            if current_price is None:
                                continue
                            
                            should_buy_result, confidence, buy_reasons = trading_logic.should_buy(ticker, current_price)
                            analyzed_count += 1
                            
                            if should_buy_result:
                                buy_candidates.append((ticker, current_price, confidence, buy_reasons))
                                
                            # 분석 진행 상황 로깅
                            if analyzed_count % 10 == 0:
                                logger.debug(f"코인 분석 진행 중... ({analyzed_count}/{len(tickers[:30])})")
                    
                    # 신뢰도 기준 정렬 후 매수 실행
                    if buy_candidates:
                        logger.info(f"매수 후보 발견: {len(buy_candidates)}개")
                        buy_candidates.sort(key=lambda x: x[2], reverse=True)
                        for ticker, price, confidence, reasons in buy_candidates[:available_slots]:
                            logger.info(f"{ticker} 매수 시도 중... (신뢰도: {confidence:.1f}%)")
                            if trading_logic.execute_buy(ticker, price, 1.0, confidence, reasons):
                                time.sleep(Settings.System.API_REQUEST_INTERVAL)
                                current_holdings += 1
                                if current_holdings >= Settings.Trading.MAX_HOLDINGS:
                                    break
                    else:
                        logger.info("적합한 매수 대상이 없습니다.")
                
                # 통계 및 모니터링
                if cycle_count % 10 == 0:  # 10 사이클마다
                    data_manager.calculate_statistics(cycle_count)
                    data_manager.monitor_performance(cycle_count)
                
                # 거래 데이터 분석 및 전략 최적화 (100 사이클마다)
                if cycle_count % 100 == 0:
                    logger.info("\n=== 거래 데이터 분석 시작 ===")
                    data_manager.analyze_trading_performance()
                    data_manager.optimize_strategy()
                    logger.info("=== 거래 데이터 분석 완료 ===\n")
                
                # API 요청 시간 업데이트
                last_request_time = time.time()
                request_count += 1
                
                # 주기적으로 요청 카운터 리셋 및 대기
                if request_count >= 100:
                    time.sleep(Settings.System.API_ERROR_INTERVAL)
                    request_count = 0
                
                logger.info(f"\n{'='*30}")
                logger.info(f"사이클 {cycle_count} 종료")
                logger.info(f"{'='*30}\n")
                
                time.sleep(settings.Trading.LOOP_INTERVAL)
                
            except Exception as e:
                logger.error(f"거래 중 오류 발생: {str(e)}")
                time.sleep(settings.Trading.ERROR_SLEEP_TIME)
                
    except KeyboardInterrupt:
        logger.warning("프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {str(e)}")
    finally:
        if emergency_stop:
            sell_all_coins()

def sell_all_coins():
    """보유한 모든 코인을 시장가로 매도"""
    try:
        # upbit 객체가 없으면 재초기화
        global upbit
        if upbit is None:
            upbit = pyupbit.Upbit(UPBIT_ACCESS, UPBIT_SECRET)
            time.sleep(0.5)

        # 보유 코인 조회
        balances = safe_api_call(upbit.get_balances)
        if not balances:
            logger.info("매도할 코인이 없습니다.")
            return

        total_coins = len([b for b in balances if b['currency'] != 'KRW'])
        if total_coins == 0:
            logger.info("매도할 코인이 없습니다.")
            return

        logger.warning(f"전체 매도 시작 - 총 {total_coins}개 코인")
        success_count = 0
        
        for balance in balances:
            if balance['currency'] == 'KRW':
                continue
                
            try:
                ticker = f"KRW-{balance['currency']}"
                amount = float(balance['balance'])
                avg_buy_price = float(balance['avg_buy_price'])
                
                # 현재가 조회
                current_price = get_current_price_safely(ticker)
                if current_price is None:
                    logger.error(f"{ticker} 현재가 조회 실패, 시장가 매도 시도")
                
                # 수익률 계산 (현재가 조회 실패시 생략)
                if current_price is not None:
                    profit_percent = ((current_price - avg_buy_price) / avg_buy_price) * 100
                    profit_message = f", 수익률: {profit_percent:.2f}%"
                else:
                    profit_message = ""
                
                # 시장가 매도
                order = safe_api_call(upbit.sell_market_order, ticker, amount)
                if order and 'error' not in order:
                    logger.info(f"{ticker} 전량 매도 완료 (수량: {amount:.8f}{profit_message})")
                    success_count += 1
                else:
                    logger.error(f"{ticker} 매도 실패: {order.get('error', '알 수 없는 오류')}")
                
                time.sleep(Settings.System.API_REQUEST_INTERVAL)
                
            except Exception as e:
                logger.error(f"{ticker} 매도 중 오류 발생: {str(e)}")
                continue
        
        # 전체 매도 결과 요약
        logger.info(f"전체 매도 완료 - 성공: {success_count}/{total_coins}")
        
        # 잔고 확인
        try:
            balance = safe_api_call(upbit.get_balance, "KRW")
            if balance is not None:
                logger.info(f"최종 KRW 잔고: {balance:,.0f}원")
        except Exception as e:
            logger.error(f"잔고 조회 실패: {str(e)}")
            
    except Exception as e:
        logger.error(f"전체 매도 중 오류 발생: {str(e)}")

def handle_signal(signum, frame):
    """시그널 핸들러"""
    global emergency_stop
    
    if signum in [signal.SIGINT, signal.SIGTERM]:
        logger.warning("프로그램 종료 신호 감지, 전체 매도를 시작합니다...")
        emergency_stop = True
        sell_all_coins()
        logger.info("프로그램이 종료되었습니다.")
        sys.exit(0)

if __name__ == "__main__":
    try:
        # 시그널 핸들러 등록
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        
        # 로깅 시스템 초기화
        setup_logging()
        logger.info("자동매매 프로그램을 시작합니다.")
        
        # 설정값 출력
        print_settings()
        
        # API 키 유효성 검사
        if not UPBIT_ACCESS or not UPBIT_SECRET:
            raise ValueError("API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
            
        # 명령행 인자 파싱
        parser = argparse.ArgumentParser(description='Upbit 자동매매 봇')
        parser.add_argument('--backtest', action='store_true', help='백테스팅 모드 실행')
        parser.add_argument('--start', type=str, help='백테스팅 시작일 (YYYY-MM-DD)')
        parser.add_argument('--end', type=str, help='백테스팅 종료일 (YYYY-MM-DD)')
        parser.add_argument('--tickers', type=str, help='테스트할 코인 티커 (콤마로 구분)')
        args = parser.parse_args()
        
        if args.backtest:
            # 백테스팅 모드
            if not args.start or not args.end:
                logger.error("백테스팅 모드는 시작일과 종료일이 필요합니다.")
                sys.exit(1)
                
            tickers = args.tickers.split(',') if args.tickers else None
            results = run_backtest_analysis(args.start, args.end, tickers)
            
        else:
            # 실제 트레이딩 모드
            main()
            
    except KeyboardInterrupt:
        logger.warning("프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {str(e)}")
    finally:
        if emergency_stop:
            sell_all_coins()

# ... existing code ...