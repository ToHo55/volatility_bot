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
import traceback
import argparse

# 터미널 색상 초기화 (자동 리셋)
init(autoreset=True)

# ===============================
# 1. 기본 설정 및 로깅 구성
# ===============================

def setup_logging():
    """로깅 시스템 설정"""
    # 기존 핸들러 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 로거 생성
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
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
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

# 로깅 시스템 초기화
logger = setup_logging()

# 환경변수 로드 및 API 키 설정
load_dotenv()
UPBIT_ACCESS = os.getenv("ACCESS_KEY")
UPBIT_SECRET = os.getenv("SECRET_KEY")

if not UPBIT_ACCESS or not UPBIT_SECRET:
    logger.error("API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    raise ValueError("API 키 오류")

try:
    upbit = pyupbit.Upbit(UPBIT_ACCESS, UPBIT_SECRET)
    balance = upbit.get_balance("KRW")
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
        BASE_TRADE_AMOUNT = 300000      # 기본 거래 금액 (원)
        MIN_TRADE_AMOUNT = 200000        # 최소 거래 금액 (원)
        COMMISSION_RATE = 0.005         # 거래 수수료 비율
        MAX_HOLDINGS = 5                # 보유 가능한 최대 코인 수
        CHECK_INTERVAL = 5              # 사이클 간격 (초)

        # 매매 전략 설정
        CONFIDENCE_THRESHOLD = 75       # 종합 신뢰도가 75% 이상이면 매수
        TAKE_PROFIT_PERCENT = 3.0       # 익절 목표: 3% 이익 발생 시
        STOP_LOSS_PERCENT = 2.0         # 손절 기준: 2% 손실 발생 시
        FORCE_SELL_MINUTES = 180        # 매수 후 180분 경과 시 강제 청산

    class Technical:
        # RSI 설정
        RSI_PERIOD = 14
        RSI_OVERSOLD = 30              # 과매도 기준값 조정
        RSI_OVERBOUGHT = 70            # 과매수 기준값 추가
        RSI_WEIGHT = 0.35              # RSI 전략 가중치 조정

        # 볼린저 밴드 설정
        BB_WINDOW = 20
        BB_STD_DEV = 2.2              # 표준편차 범위 확대
        BB_WEIGHT = 0.35              # BB 전략 가중치 조정

        # 변동성 돌파 설정
        VOLATILITY_K = 0.4            # 변동성 계수 조정 (보수적으로)
        VOLATILITY_WEIGHT = 0.2       # 변동성 돌파 전략 가중치 조정

        # 이동평균선 설정
        MA_SHORT = 10                 # 단기 이동평균선 기간 조정
        MA_LONG = 30                  # 장기 이동평균선 기간 조정
        MA_WEIGHT = 0.1              # 이동평균선 전략 가중치 조정

        # 추가: 트레일링 스탑 설정
        TRAILING_STOP_START = 1.5     # 수익률이 1.5% 이상일 때 트레일링 스탑 시작
        TRAILING_STOP_DISTANCE = 1.0  # 최고점 대비 1.0% 하락 시 매도

    class Market:
        # 거래량 필터링
        MIN_TRADE_VOLUME = 100000        # 최소 거래량 조건 (5분봉 기준)
        VOLUME_UPDATE_INTERVAL = 30     # 거래량 필터링 주기 (분)
        
        # 제외 코인
        STABLE_COINS = ["KRW-USDT", "KRW-DAI", "KRW-TUSD", "KRW-BUSD", "KRW-USDC"]

    class System:
        # API 요청 관리
        API_REQUEST_INTERVAL = 0.1      # API 요청 간격
        CACHE_EXPIRY = 300             # 캐시 만료 시간 (초)
        
        # 로그 설정
        LOG_ROTATION_SIZE = 10485760    # 로그 파일 최대 크기 (10MB)
        LOG_BACKUP_COUNT = 5            # 보관할 로그 파일 수

# 캐시 관리
class Cache:
    def __init__(self):
        self.cache = {}
        self.cache_dir = "cache"
        self.ensure_cache_dir()
        
    def ensure_cache_dir(self):
        """캐시 디렉토리 생성"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
    def get_cache_path(self, key: str) -> str:
        """캐시 파일 경로 반환"""
        return os.path.join(self.cache_dir, f"{key}.pkl")
    
    def get(self, key: str, max_age_seconds: int = 300):
        """캐시된 데이터 조회"""
        try:
            # 메모리 캐시 확인
            if key in self.cache:
                data, timestamp = self.cache[key]
                if time.time() - timestamp <= max_age_seconds:
                    return data
                    
            # 파일 캐시 확인
            cache_path = self.get_cache_path(key)
            if os.path.exists(cache_path):
                modified_time = os.path.getmtime(cache_path)
                if time.time() - modified_time <= max_age_seconds:
                    with open(cache_path, 'rb') as f:
                        data = pd.read_pickle(f)
                        self.cache[key] = (data, time.time())
                        return data
        except Exception as e:
            logging.warning(f"캐시 조회 중 오류: {str(e)}")
        return None
        
    def set(self, key: str, value: any):
        """데이터 캐싱"""
        try:
            # 메모리 캐시 저장
            self.cache[key] = (value, time.time())
            
            # 파일 캐시 저장
            cache_path = self.get_cache_path(key)
            with open(cache_path, 'wb') as f:
                pd.to_pickle(value, f)
        except Exception as e:
            logging.error(f"캐시 저장 중 오류: {str(e)}")
            
    def clear(self, max_age_seconds: int = None):
        """캐시 정리"""
        try:
            current_time = time.time()
            
            # 메모리 캐시 정리
            if max_age_seconds:
                self.cache = {
                    k: (v, t) for k, (v, t) in self.cache.items()
                    if current_time - t <= max_age_seconds
                }
            else:
                self.cache.clear()
            
            # 파일 캐시 정리
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if max_age_seconds:
                    modified_time = os.path.getmtime(file_path)
                    if current_time - modified_time > max_age_seconds:
                        os.remove(file_path)
                else:
                    os.remove(file_path)
        except Exception as e:
            logging.error(f"캐시 정리 중 오류: {str(e)}")

# 캐시 인스턴스 생성
ohlcv_cache = Cache()
indicator_cache = Cache()

# ===============================
# 전역 변수: 보유 코인 및 거래 기록
# ===============================
held_coins = {}  # {ticker: {"buy_price": float, "buy_time": datetime}}
trade_history = []  # 거래 내역을 JSON 파일에 저장할 예정
trading_logic = None  # TradingLogic 인스턴스를 저장할 전역 변수

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
current_cycle_counts = {
    'buy': 0,
    'tp': 0,
    'sl_normal': 0,
    'sl_forced': 0,
    'sl_other': 0,
    'error': 0
}
cumulative_profit = 1.0

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
    filtered = []
    for ticker in markets:
        df = safe_api_call(pyupbit.get_ohlcv, ticker, interval="minute5", count=100)
        if df is None or df.empty:
            continue
        avg_volume = df['volume'].mean()
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
def calculate_rsi(prices, period):
    if len(prices) < period + 1:
        return None
    deltas = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
    gains = [delta if delta > 0 else 0 for delta in deltas]
    losses = [-delta if delta < 0 else 0 for delta in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bb_position(df, window, std_dev):
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    upper_band = rolling_mean + std_dev * rolling_std
    lower_band = rolling_mean - std_dev * rolling_std
    current_price = df['close'].iloc[-1]
    if upper_band.iloc[-1] - lower_band.iloc[-1] == 0:
        return 50.0
    position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
    return position * 100  # 백분율

def calculate_confidence(ticker):
    """
    개별 코인마다 RSI와 Bollinger Band 기반 신뢰도를 계산
    - RSI 신뢰도: RSI 값에 따른 가중치 부여
    - BB 신뢰도: 볼린저 밴드 위치와 추세를 고려
    - 거래량 신뢰도: 최근 거래량 증가율 고려
    """
    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=100)
    time.sleep(Settings.System.API_REQUEST_INTERVAL)
    
    if df is None or df.empty or len(df) < Settings.Technical.BB_WINDOW:
        return 0, None, None
        
    try:
        # RSI 신뢰도 계산
        close_prices = df['close'].tolist()
        rsi = calculate_rsi(close_prices, Settings.Technical.RSI_PERIOD)
        
        if rsi is None:
            return 0, None, None
            
        # RSI 신뢰도: 과매도 구간에서 높은 점수, 과매수 구간에서 낮은 점수
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
            
        # 볼린저 밴드 신뢰도 계산
        bb_pos = calculate_bb_position(df, Settings.Technical.BB_WINDOW, Settings.Technical.BB_STD_DEV)
        
        # BB 위치에 따른 신뢰도
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
            
        # 거래량 신뢰도 계산
        recent_volume = df['volume'].iloc[-5:].mean()  # 최근 5개 봉 평균
        prev_volume = df['volume'].iloc[-10:-5].mean()  # 이전 5개 봉 평균
        
        volume_ratio = recent_volume / prev_volume if prev_volume > 0 else 1
        volume_conf = min(100, max(0, 50 * (volume_ratio - 0.5)))  # 거래량 증가율에 따른 신뢰도
        
        # 최종 신뢰도 계산 (가중 평균)
        # RSI: 40%, BB: 40%, 거래량: 20%
        confidence = (rsi_conf * Settings.Technical.RSI_WEIGHT) + (bb_conf * Settings.Technical.BB_WEIGHT) + (volume_conf * 0.2)
        
        logger.info(f"{ticker} 신뢰도 분석:")
        logger.info(f"- RSI({rsi:.1f}): {rsi_conf:.1f}점")
        logger.info(f"- BB({bb_pos:.1f}%): {bb_conf:.1f}점")
        logger.info(f"- Volume(비율: {volume_ratio:.2f}): {volume_conf:.1f}점")
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
    def __init__(self):
        self.atr_period = 14
        self.atr_multiplier_tp = 1.5
        self.atr_multiplier_sl = 0.7
        self.trailing_activation = 0.05
        self.trailing_stop_percent = 0.02
        self.held_coins = {}
        self.data_manager = DataManager()
        self.strategy_composer = StrategyComposer()
        self.load_held_coins()

    def execute_buy(self, ticker, current_price, market_weight, confidence):
        """매수 실행"""
        try:
            # 매수 금액 계산
            available_balance = upbit.get_balance("KRW")
            if available_balance < Settings.Trading.MIN_TRADE_AMOUNT:
                logger.warning(f"{ticker} 매수 불가 - 최소 주문 금액({Settings.Trading.MIN_TRADE_AMOUNT:,}원) 미달")
                return False

            # 최대 매수 가능 금액 계산
            trade_amount = min(
                Settings.Trading.BASE_TRADE_AMOUNT,
                available_balance * 0.9995  # 수수료 고려
            )

            # 매수 주문
            order = upbit.buy_market_order(ticker, trade_amount)
            if not order or 'error' in order:
                logger.error(f"{ticker} 매수 주문 실패")
                return False

            # 주문 상태 확인
            time.sleep(0.5)
            order_detail = upbit.get_order(order['uuid'])
            if not order_detail:
                logger.error(f"{ticker} 주문 상태 확인 실패")
                return False

            executed_volume = float(order_detail['executed_volume'])
            if executed_volume <= 0:
                logger.error(f"{ticker} 매수 실패 - 체결 수량이 0입니다")
                return False

            executed_price = float(order_detail['price']) / executed_volume
            logger.info(f"{ticker} 매수 성공 - 체결단가: {executed_price:,.2f}원, 수량: {executed_volume:.8f}")

            # 보유 코인 정보 업데이트
            self.held_coins[ticker] = {
                'amount': executed_volume,
                'buy_price': executed_price,
                'buy_time': datetime.now(KST),
                'confidence': confidence
            }

            return True

        except Exception as e:
            logger.error(f"{ticker} 매수 실행 중 오류: {str(e)}")
            return False

    def execute_sell(self, ticker, current_price, reason=''):
        """매도 실행 함수"""
        try:
            if ticker not in self.held_coins:
                logger.error(f"{ticker} 매도 실패 - 보유 정보 없음")
                return False

            amount = self.held_coins[ticker]['amount']
            buy_price = self.held_coins[ticker]['buy_price']
            
            order = upbit.sell_market_order(ticker, amount)
            if not order or 'error' in order:
                logger.error(f"{ticker} 매도 주문 실패")
                return False

            time.sleep(0.5)
            order_detail = upbit.get_order(order['uuid'])
            if not order_detail:
                logger.error(f"{ticker} 매도 주문 상태 확인 실패")
                return False

            executed_price = float(order_detail['price']) / float(order_detail['executed_volume'])
            profit_loss = ((executed_price - buy_price) / buy_price) * 100
            
            # 거래 데이터 로깅
            trade_data = {
                "timestamp": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
                "cycle_num": cycle_count,
                "ticker": ticker,
                "action": "sell",
                "price": executed_price,
                "atr_value": self.calculate_atr(ticker),
                "profit_loss": profit_loss,
                "holding_time": self.calculate_holding_time(ticker),
                "sell_reason": reason,
                "market_condition": self.get_market_condition(ticker)
            }
            self.data_manager.log_trade_data(trade_data)
            
            # 보유 코인 정보 삭제
            del self.held_coins[ticker]
            
            logger.info(f"{ticker} 매도 성공 - 체결단가: {executed_price:,.2f}원, 수익률: {profit_loss:.2f}%")
            return True
            
        except Exception as e:
            logger.error(f"{ticker} 매도 실행 중 오류: {str(e)}")
            return False

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
            balances = upbit.get_balances()
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

    def should_buy(self, ticker, current_price):
        """매수 여부 결정"""
        try:
            # 이미 보유 중인 코인인 경우 추가 매수 여부 확인
            if ticker in self.held_coins:
                buy_price = self.held_coins[ticker]["buy_price"]
                # 현재가가 매수 평균가보다 높으면 추가 매수하지 않음
                if current_price >= buy_price:
                    return False, 0
            
            # OHLCV 데이터 조회
            df = get_ohlcv_cached(ticker, interval="minute5", count=100)
            if df is None or df.empty:
                return False, 0
            
            # 전략 분석 실행
            analysis = self.strategy_composer.analyze(ticker, df, current_price)
            if not analysis:
                return False, 0
            
            total_confidence = analysis['total_confidence']
            
            # 신뢰도가 기준값 이상이면 매수 신호
            return total_confidence >= Settings.Trading.CONFIDENCE_THRESHOLD, total_confidence
            
        except Exception as e:
            logger.error(f"{ticker} 매수 신호 계산 중 오류: {str(e)}")
            return False, 0

    def update_trailing_stop(self, ticker, current_price):
        """트레일링 스탑 업데이트"""
        if ticker not in self.held_coins:
            return None

        coin_data = self.held_coins[ticker]
        max_price = coin_data.get("max_price", current_price)
        
        if current_price > max_price:
            max_price = current_price
            coin_data["max_price"] = max_price
        
        trailing_stop_price = max_price * (1 - self.trailing_stop_percent)
        return trailing_stop_price

    def get_dynamic_targets(self, ticker, current_price):
        """ATR 기반 동적 목표가 설정"""
        atr = self.calculate_atr(ticker)
        if atr is None:
            return None, None

        take_profit = current_price + (atr * self.atr_multiplier_tp)
        stop_loss = current_price - (atr * self.atr_multiplier_sl)
        
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

    def load_held_coins(self):
        """보유 중인 코인 정보 로드"""
        try:
            balances = upbit.get_balances()
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

    def execute_trade_decision(self, ticker, current_price, market_weight, confidence):
        """매매 결정을 실행하는 메서드"""
        try:
            # 매수 조건 확인
            if self.should_buy(ticker, current_price):
                return self.execute_buy(ticker, current_price, market_weight, confidence)
            
            # 보유 중인 코인이면 매도 조건 확인
            elif ticker in held_coins:
                holding_time = self.calculate_holding_time(ticker)
                buy_price = held_coins[ticker]["buy_price"]
                
                # 익절 조건
                if current_price >= buy_price * (1 + Settings.Trading.TAKE_PROFIT_PERCENT / 100):
                    return self.execute_sell(ticker, current_price, reason='tp')
                
                # 손절 조건
                elif current_price <= buy_price * (1 - Settings.Trading.STOP_LOSS_PERCENT / 100):
                    return self.execute_sell(ticker, current_price, reason='sl')
                
                # 강제 청산 조건
                elif holding_time >= Settings.Trading.FORCE_SELL_MINUTES:
                    return self.execute_sell(ticker, current_price, reason='force')
            
            return False
            
        except Exception as e:
            logger.error(f"{ticker} 매매 결정 실행 중 오류: {str(e)}")
            return False

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
    if cached is not None:
        return cached

    df = safe_api_call(pyupbit.get_ohlcv, ticker, interval=interval, count=count)
    if df is not None:
        ohlcv_cache.set(cache_key, df)
    return df

def print_settings():
    """현재 설정값들을 로깅"""
    logger.info("====== 현재 설정값 ======")
    logger.info("[거래 설정]")
    logger.info(f"기본 거래금액: {Settings.Trading.BASE_TRADE_AMOUNT:,}원")
    logger.info(f"최소 거래금액: {Settings.Trading.MIN_TRADE_AMOUNT:,}원")
    logger.info(f"최대 보유 코인수: {Settings.Trading.MAX_HOLDINGS}개")
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

class TradingStrategy:
    """기본 트레이딩 전략 인터페이스"""
    def analyze(self, ticker, df, current_price):
        raise NotImplementedError
        
    def calculate_confidence(self, analysis_result):
        raise NotImplementedError

class RSIBBStrategy(TradingStrategy):
    """RSI와 볼린저밴드 기반 전략"""
    def analyze(self, ticker, df, current_price):
        """RSI와 볼린저밴드 기반 전략"""
        try:
            if df is None or len(df) < max(Settings.Technical.RSI_PERIOD, Settings.Technical.BB_WINDOW):
                return None

            # RSI 계산
            rsi = Indicators.calculate_rsi(df, Settings.Technical.RSI_PERIOD)
            if rsi is None:
                return None

            # 볼린저 밴드 계산
            upper_band, lower_band, bb_pos = Indicators.calculate_bollinger_bands(
                df, 
                Settings.Technical.BB_WINDOW,
                Settings.Technical.BB_STD_DEV
            )
            if bb_pos is None:
                return None

            # 이동평균선 기울기 계산
            ma_short = df['close'].rolling(window=Settings.Technical.MA_SHORT).mean()
            ma_trend = (ma_short.iloc[-1] - ma_short.iloc[-2]) / ma_short.iloc[-2] * 100

            return {
                'rsi': rsi,
                'bb_position': bb_pos,
                'ma_trend': ma_trend,
                'strategy': 'RSI_BB',
                'current_price': current_price
            }
            
        except Exception as e:
            logging.error(f"RSI/BB 분석 중 오류: {str(e)}")
            return None

    def calculate_confidence(self, analysis_result):
        """신뢰도 계산"""
        try:
            if analysis_result is None:
                return 0

            rsi = analysis_result.get('rsi', 0)
            bb_position = analysis_result.get('bb_position', 50)
            ma_trend = analysis_result.get('ma_trend', 0)
            
            # RSI 신뢰도 계산 (과매도 구간에서 더 높은 신뢰도)
            if rsi <= 30:
                rsi_confidence = 90 + (30 - rsi)  # 20-30: 90-100점
            elif rsi <= 40:
                rsi_confidence = 70 + (40 - rsi) * 2  # 30-40: 70-90점
            elif rsi <= 50:
                rsi_confidence = 50 + (50 - rsi) * 2  # 40-50: 50-70점
            else:
                rsi_confidence = max(0, 100 - rsi)  # RSI가 높을수록 신뢰도 감소
            
            # 볼린저 밴드 신뢰도 계산 (하단 밴드 근처에서 더 높은 신뢰도)
            if bb_position <= 20:  # 하단 밴드 근처
                bb_confidence = 90 + (20 - bb_position)
            elif bb_position <= 40:
                bb_confidence = 70 + (40 - bb_position)
            elif bb_position <= 60:
                bb_confidence = 50
            else:
                bb_confidence = max(0, 100 - bb_position)
            
            # 이동평균선 기울기 확인
            trend_confidence = 60 if ma_trend > 0 else 40
            
            # 최종 신뢰도 계산 (가중 평균)
            weights = [
                (rsi_confidence, Settings.Technical.RSI_WEIGHT),
                (bb_confidence, Settings.Technical.BB_WEIGHT),
                (trend_confidence, Settings.Technical.MA_WEIGHT)
            ]
            
            total_weight = sum(w for _, w in weights)
            final_confidence = sum(conf * weight for conf, weight in weights) / total_weight
            
            logging.debug(f"신뢰도 분석 - RSI: {rsi_confidence:.1f}, BB: {bb_confidence:.1f}, Trend: {trend_confidence:.1f}, Final: {final_confidence:.1f}")
            
            return final_confidence
            
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
        
        # 목표가 돌파 시 신뢰도 상승
        if current > target:
            # 변동성이 0인 경우 처리
            if volatility == 0:
                confidence = 50  # 기본값 설정
            else:
                # 돌파 정도에 따른 신뢰도 계산
                breakout_ratio = (current - target) / volatility
                confidence = min(100, 50 + breakout_ratio * 50)
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

class MACDStrategy(TradingStrategy):
    """MACD(Moving Average Convergence Divergence) 전략"""
    
    def analyze(self, ticker, df, current_price):
        try:
            if df is None or len(df) < 26:  # MACD 계산에 필요한 최소 데이터
                return None

            # MACD 계산
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - signal
            
            if len(macd_hist) < 2:
                return None

            # 현재 MACD 상태
            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]
            current_hist = macd_hist.iloc[-1]
            prev_hist = macd_hist.iloc[-2]
            
            # MACD 크로스 및 히스토그램 분석
            is_golden_cross = current_macd > current_signal and macd.iloc[-2] <= signal.iloc[-2]
            is_death_cross = current_macd < current_signal and macd.iloc[-2] >= signal.iloc[-2]
            hist_momentum = current_hist > prev_hist
            
            return {
                'macd': current_macd,
                'signal': current_signal,
                'histogram': current_hist,
                'is_golden_cross': is_golden_cross,
                'is_death_cross': is_death_cross,
                'hist_momentum': hist_momentum,
                'strategy': 'MACD'
            }
            
        except Exception as e:
            logging.error(f"MACD 전략 분석 중 오류: {str(e)}")
            return None

    def calculate_confidence(self, analysis_result):
        try:
            if analysis_result is None:
                return 0

            confidence = 50  # 기본 신뢰도
            
            # 골든크로스/데드크로스 영향
            if analysis_result['is_golden_cross']:
                confidence += 30
            elif analysis_result['is_death_cross']:
                confidence -= 30
            
            # 히스토그램 모멘텀 영향
            if analysis_result['hist_momentum']:
                confidence += 10
            else:
                confidence -= 10
                
            # MACD와 시그널 라인의 거리 기반 가중치
            macd_signal_gap = abs(analysis_result['macd'] - analysis_result['signal'])
            if macd_signal_gap > 0:
                confidence += min(10, macd_signal_gap * 5)
                
            return max(0, min(100, confidence))
            
        except Exception as e:
            logging.error(f"MACD 신뢰도 계산 중 오류: {str(e)}")
            return 0

class StrategyComposer:
    """전략 조합기"""
    def __init__(self):
        self.strategies = {
            'rsi_bb': RSIBBStrategy(),
            'volatility': VolatilityBreakoutStrategy(),
            'ma': MAStrategy(),
            'macd': MACDStrategy()  # MACD 전략 추가
        }
        
    def analyze(self, ticker, df, current_price):
        results = {}
        confidences = []
        
        for name, strategy in self.strategies.items():
            try:
                analysis = strategy.analyze(ticker, df, current_price)
                confidence = strategy.calculate_confidence(analysis)
                results[name] = {'analysis': analysis, 'confidence': confidence}
                confidences.append(confidence)
            except Exception as e:
                logging.error(f"{name} 전략 분석 중 오류: {str(e)}")
                results[name] = {'analysis': None, 'confidence': 0}
                confidences.append(0)
        
        # 전체 신뢰도 계산 (가중 평균)
        weights = {
            'rsi_bb': Settings.Technical.RSI_WEIGHT + Settings.Technical.BB_WEIGHT,
            'volatility': Settings.Technical.VOLATILITY_WEIGHT,
            'ma': Settings.Technical.MA_WEIGHT,
            'macd': 0.2  # MACD 전략 가중치
        }
        
        total_confidence = 0
        total_weight = sum(weights.values())
        
        for name, weight in weights.items():
            total_confidence += results[name]['confidence'] * (weight / total_weight)
        
        results['total_confidence'] = round(total_confidence, 2)
        return results

class DataManager:
    def __init__(self):
        self.base_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 디렉토리
        self.log_file = self.setup_data_logging()
        self.last_stats_cycle = 0
        
    def setup_data_logging(self):
        """데이터 로깅 설정"""
        try:
            # 로그 파일 경로 설정
            current_time = datetime.now(KST).strftime("%Y%m%d_%H%M")
            log_filename = f"trading_data_{current_time}.csv"
            log_file_path = os.path.join(self.base_path, log_filename)
            
            # 파일 생성
            headers = [
                "timestamp", "cycle_num", 
                "ticker", "action", 
                "price", "atr_value",
                "profit_loss", "holding_time",
                "sell_reason", "market_condition"
            ]
            
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
                        "price", "atr_value",
                        "profit_loss", "holding_time",
                        "sell_reason", "market_condition"
                    ]
                ]
                writer.writerow(row)
                logger.debug(f"거래 데이터 기록 완료: {data_dict['ticker']} - {data_dict['action']}")
        except Exception as e:
            logger.error(f"데이터 로깅 중 오류 발생: {str(e)}")
            
    def calculate_statistics(self, current_cycle):
        """주기적 통계 계산 (100 사이클마다)"""
        if current_cycle % 100 != 0 or current_cycle == self.last_stats_cycle:
            return
            
        try:
            df = pd.read_csv(self.log_file)
            cycle_start = max(0, current_cycle - 100)
            cycle_data = df[(df['cycle_num'] >= cycle_start) & (df['cycle_num'] <= current_cycle)]
            
            if len(cycle_data) == 0:
                return
                
            stats = {
                "기간": f"사이클 {cycle_start}-{current_cycle}",
                "총 거래수": len(cycle_data),
                "승률": f"{(cycle_data['profit_loss'] > 0).mean() * 100:.2f}%",
                "평균수익률": f"{cycle_data['profit_loss'].mean():.2f}%",
                "최대손실": f"{cycle_data['profit_loss'].min():.2f}%",
                "최대수익": f"{cycle_data['profit_loss'].max():.2f}%",
                "평균보유시간": f"{cycle_data['holding_time'].mean():.2f}분"
            }
            
            logger.info("\n=== 거래 통계 요약 ===")
            for key, value in stats.items():
                logger.info(f"{key}: {value}")
            logger.info("=====================")
            
            self.last_stats_cycle = current_cycle
            
        except Exception as e:
            logger.error(f"통계 계산 중 오류: {str(e)}")
            
    def monitor_performance(self, current_cycle):
        """실시간 성능 모니터링 (10 사이클마다)"""
        if current_cycle % 10 != 0:
            return
            
        try:
            df = pd.read_csv(self.log_file)
            recent_data = df.tail(100)  # 최근 100개 거래
            
            if len(recent_data) == 0:
                return
                
            recent_stats = {
                "최근 거래 수": len(recent_data),
                "최근 승률": f"{(recent_data['profit_loss'] > 0).mean() * 100:.2f}%",
                "최근 평균수익": f"{recent_data['profit_loss'].mean():.2f}%",
                "ATR 평균": f"{recent_data['atr_value'].mean():.2f}"
            }
            
            logger.info("\n=== 실시간 성능 지표 ===")
            for key, value in recent_stats.items():
                logger.info(f"{key}: {value}")
            logger.info("=====================")
            
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

# -------------------------------
# 전역 변수 및 유틸리티 함수
# -------------------------------
emergency_stop = False
cycle_count = 0
cumulative_counts = {
    'buy': 0,
    'tp': 0,
    'sl_normal': 0,
    'sl_forced': 0,
    'sl_other': 0,
    'error': 0
}
current_cycle_counts = {
    'buy': 0,
    'tp': 0,
    'sl_normal': 0,
    'sl_forced': 0,
    'sl_other': 0,
    'error': 0
}

def safe_api_call(func, *args, **kwargs):
    """API 호출 안전하게 처리"""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"API 호출 실패 (최대 재시도 횟수 초과): {str(e)}")
                raise
            logging.warning(f"API 호출 실패 (재시도 {attempt + 1}/{max_retries}): {str(e)}")
            time.sleep(retry_delay * (attempt + 1))

def check_data_validity(df: pd.DataFrame) -> bool:
    """데이터 유효성 검사"""
    if df is None or df.empty:
        return False
        
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        return False
        
    if df.isnull().any().any():
        return False
        
    return True

def sell_all_coins():
    """보유한 모든 코인을 시장가로 매도"""
    try:
        balances = upbit.get_balances()
        if not balances:
            logger.warning("매도할 코인이 없습니다.")
            return

        success_count = 0
        for balance in balances:
            if balance['currency'] == 'KRW':
                continue
            
            ticker = f"KRW-{balance['currency']}"
            amount = float(balance['balance'])
            
            if amount > 0:
                try:
                    result = upbit.sell_market_order(ticker, amount)
                    if result and 'error' not in result:
                        success_count += 1
                        logger.info(f"{ticker} 전량 매도 완료")
                    else:
                        logger.error(f"{ticker} 매도 실패: {result}")
                except Exception as e:
                    logger.error(f"{ticker} 매도 중 오류 발생: {str(e)}")
                
                time.sleep(0.1)  # API 호출 간격 준수

        logger.info(f"전체 매도 완료: {success_count}개 코인 매도됨")
        
    except Exception as e:
        logger.error(f"전체 매도 중 오류 발생: {str(e)}")
    finally:
        if emergency_stop:
            logger.info("프로그램을 종료합니다.")
            sys.exit(0)

def main():
    """메인 실행 함수"""
    try:
        # 명령행 인자 파싱
        parser = argparse.ArgumentParser(description='백테스팅 설정')
        parser.add_argument('--start_date', required=True, help='시작일 (YYYY-MM-DD)')
        parser.add_argument('--end_date', required=True, help='종료일 (YYYY-MM-DD)')
        parser.add_argument('--initial_balance', type=int, default=10000000, help='초기 자본금')
        args = parser.parse_args()

        # 로깅 설정
        setup_logging()
        
        # 초기 설정 출력
        logging.info(f"초기 KRW 잔고: {args.initial_balance:,}원")
        logging.info("백테스트를 시작합니다.")
        print_settings()
        
        # 백테스팅 실행
        backtester = Backtesting(args.start_date, args.end_date, args.initial_balance)
        backtester.run_backtest()
        
    except KeyboardInterrupt:
        logging.info("\n사용자에 의해 프로그램이 중단되었습니다.")
    except Exception as e:
        logging.error(f"프로그램 실행 중 오류 발생: {str(e)}")
        traceback.print_exc()
    finally:
        logging.info("프로그램이 종료되었습니다.")

class Backtesting:
    def __init__(self, start_date, end_date, initial_balance=10000000):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.holdings = {}
        self.trades = []
        self.daily_balance = {}
        self.cache = Cache()
        
        # 전략 초기화
        self.strategy_composer = StrategyComposer()
        
        # 데이터 관리자 초기화
        self.data_manager = DataManager()
        
    def check_volume_requirement(self, ticker, current_time):
        """거래량 요구사항 확인"""
        try:
            # 캐시에서 데이터 조회
            df = self.cache.get(f"price_{ticker}")
            if df is None:
                return False
            
            # 현재 시점 이전의 데이터만 필터링
            mask = df.index <= current_time
            filtered_df = df[mask]
            
            if filtered_df.empty:
                return False
                
            # 최근 거래량 평균 계산
            recent_volume = filtered_df['volume'].iloc[-1]
            return recent_volume >= Settings.Market.MIN_TRADE_VOLUME
            
        except Exception as e:
            logging.error(f"{ticker} 거래량 확인 중 오류: {str(e)}")
            return False

    def record_daily_balance(self, date):
        """일별 잔고 기록"""
        try:
            date_key = date.date()
            self.daily_balance[date_key] = {
                'balance': self.current_balance,
                'holdings': len(self.holdings),
                'profit_rate': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
            }
        except Exception as e:
            logging.error(f"일별 잔고 기록 중 오류 발생: {str(e)}")
            traceback.print_exc()

    def run_backtest(self, tickers=None):
        """백테스트 실행"""
        if tickers is None:
            tickers = self.load_tickers()
        
        # 모든 티커의 가격 데이터를 미리 로드
        logging.info("가격 데이터 로드 중...")
        loaded_count = 0
        cached_count = 0
        for ticker in tickers:
            df = self.load_price_data(ticker)
            if df is not None and not df.empty:
                loaded_count += 1
                cached_data = self.cache.get(f"price_{ticker}")
                if cached_data is not None and not cached_data.empty:
                    cached_count += 1
        logging.info(f"가격 데이터 로드 완료 (총 {loaded_count}개, 캐시 사용 {cached_count}개)")
        
        # 백테스팅 실행
        current_date = self.start_date
        while current_date <= self.end_date:
            if not self.simulate_day(current_date, tickers):
                break
            self.record_daily_balance(current_date)
            current_date += timedelta(days=1)
        
        # 결과 출력 및 저장
        self.print_backtest_results()
        self.plot_results()
        self.save_results()

    def simulate_day(self, date, tickers):
        """하루 동안의 거래 시뮬레이션"""
        try:
            # 초기 자본금 대비 손실률 계산
            current_loss_rate = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
            
            # 손실률이 -10%를 초과하면 백테스팅 종료
            if current_loss_rate <= -10:
                logging.warning(f"손실률 {current_loss_rate:.2f}%가 -10%를 초과하여 백테스팅을 종료합니다.")
                return False
            
            # 수익률이 30%를 초과하면 백테스팅 종료
            current_profit_rate = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
            if current_profit_rate >= 30:
                logging.info(f"목표 수익률 {current_profit_rate:.2f}%를 달성하여 백테스팅을 종료합니다.")
                return False
            
            logging.info(f"\n=== {date.date()} 거래 시작 ===")
            
            # 5분 단위로 시뮬레이션
            for minute in range(0, 24*60, 5):
                current_time = date + timedelta(minutes=minute)
                
                # 보유 코인 점검
                self.check_holdings(current_time)
                
                # 새로운 매수 기회 탐색
                if len(self.holdings) < Settings.Trading.MAX_HOLDINGS:
                    opportunities = self.find_buying_opportunities(current_time, tickers)
                    for opp in opportunities:
                        self.execute_buy(opp['ticker'], opp['price'], current_time, opp['confidence'])
                
                # 매 시간마다 상태 출력
                if minute % 60 == 0:
                    logging.info(f"시간: {current_time.strftime('%H:%M')} - 현재 잔고: {self.current_balance:,.0f}원, 보유 코인: {len(self.holdings)}개")

            # 하루 종료 시 거래 요약 출력
            self.print_daily_summary(date)
            logging.info(f"=== {date.date()} 거래 종료 ===\n")
            return True

        except Exception as e:
            logging.error(f"시뮬레이션 중 오류 발생: {str(e)}")
            traceback.print_exc()
            return False

    def load_tickers(self):
        """거래 대상 코인 목록 로드"""
        try:
            # KRW 마켓의 모든 코인 (스테이블 코인 제외)
            tickers = [ticker for ticker in pyupbit.get_tickers(fiat="KRW") 
                      if ticker not in Settings.Market.STABLE_COINS]
            logging.info(f"거래 대상 코인 목록 로드 완료: {len(tickers)}개")
            return tickers
        except Exception as e:
            logging.error(f"거래 대상 코인 목록 로드 중 오류: {str(e)}")
            return []

    def load_price_data(self, ticker):
        """가격 데이터 로드"""
        try:
            # 캐시에서 먼저 확인
            cached_data = self.cache.get(f"price_{ticker}")
            if cached_data is not None and not cached_data.empty:
                return cached_data

            # 데이터 로드 시도
            df = None
            retry_count = 3
            for attempt in range(retry_count):
                try:
                    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=1000)
                    if df is not None and not df.empty:
                        break
                    time.sleep(0.1)
                except Exception as e:
                    logging.warning(f"{ticker} 데이터 로드 재시도 {attempt + 1}/{retry_count}: {str(e)}")
                    time.sleep(0.5)

            if df is None or df.empty:
                logging.error(f"{ticker} 데이터 로드 실패")
                return None

            # 인덱스가 datetime 형식인지 확인하고 변환
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # 백테스팅 기간에 맞게 필터링
            df = df.loc[self.start_date:self.end_date]

            if df.empty:
                logging.warning(f"{ticker} 해당 기간의 데이터가 없습니다")
                return None

            # 캐시에 저장
            self.cache.set(f"price_{ticker}", df)
            
            logging.info(f"{ticker} 데이터 로드 성공: {len(df)}개 데이터")
            return df
            
        except Exception as e:
            logging.error(f"{ticker} 데이터 로드 중 오류: {str(e)}")
            return None

    def check_holdings(self, current_time):
        """보유 코인 점검"""
        for ticker, data in list(self.holdings.items()):
            try:
                df = self.cache.get(f"price_{ticker}")
                if df is None:
                    continue

                # 현재 시점까지의 데이터만 사용
                mask = df.index <= current_time
                current_data = df[mask].copy()  # copy() 추가
                
                if current_data.empty:
                    continue
                    
                current_price = current_data['close'].iloc[-1]
                buy_price = data['buy_price']
                
                # 매수 시간이 없는 경우 현재 시간으로 설정
                if 'buy_time' not in data:
                    data['buy_time'] = current_time
                    
                holding_time = (current_time - data['buy_time']).total_seconds() / 60
                
                # 익절, 손절, 강제 청산 조건 확인
                profit_rate = ((current_price - buy_price) / buy_price) * 100
                
                if profit_rate >= Settings.Trading.TAKE_PROFIT_PERCENT:
                    self.execute_sell(ticker, current_price, current_time, 'take_profit')
                elif profit_rate <= -Settings.Trading.STOP_LOSS_PERCENT:
                    self.execute_sell(ticker, current_price, current_time, 'stop_loss')
                elif holding_time >= Settings.Trading.FORCE_SELL_MINUTES:
                    self.execute_sell(ticker, current_price, current_time, 'force_sell')
                    
            except Exception as e:
                logging.error(f"{ticker} 보유 코인 점검 중 오류: {str(e)}")

    def find_buying_opportunities(self, current_time, tickers):
        """매수 기회 탐색"""
        opportunities = []
        
        for ticker in tickers:
            try:
                if ticker in self.holdings:
                    continue
                    
                df = self.cache.get(f"price_{ticker}")
                if df is None:
                    continue
                    
                # 현재 시점까지의 데이터만 사용
                mask = df.index <= current_time
                current_data = df[mask].copy()  # copy() 추가
                
                if len(current_data) < 30:  # 최소 데이터 포인트 확인
                    continue
                    
                current_price = current_data['close'].iloc[-1]
                
                # 거래량 요구사항 확인
                if not self.check_volume_requirement(ticker, current_time):
                    continue
                    
                # 전략 분석
                analysis = self.strategy_composer.analyze(ticker, current_data, current_price)
                if analysis and analysis.get('total_confidence', 0) >= Settings.Trading.CONFIDENCE_THRESHOLD:
                    opportunities.append({
                        'ticker': ticker,
                        'price': current_price,
                        'confidence': analysis['total_confidence']
                    })
                    
            except Exception as e:
                logging.error(f"{ticker} 매수 기회 분석 중 오류: {str(e)}")
                continue
                
        return opportunities

    def execute_buy(self, ticker, price, current_time, confidence):
        """매수 실행"""
        try:
            # 매수 가능 금액 계산
            available_amount = min(
                self.current_balance * 0.9995,  # 수수료 고려
                Settings.Trading.BASE_TRADE_AMOUNT
            )
            
            if available_amount < Settings.Trading.MIN_TRADE_AMOUNT:
                return False
                
            # 매수 수량 계산
            quantity = available_amount / price
            total_cost = quantity * price * (1 + Settings.Trading.COMMISSION_RATE)
            
            # 잔고 업데이트
            self.current_balance -= total_cost
            
            # 보유 코인 정보 업데이트
            self.holdings[ticker] = {
                'amount': quantity,
                'buy_price': price,
                'buy_time': current_time,
                'confidence': confidence
            }
            
            # 거래 기록
            self.trades.append({
                'timestamp': current_time,
                'ticker': ticker,
                'action': 'buy',
                'price': price,
                'quantity': quantity,
                'confidence': confidence
            })
            
            logging.info(f"{ticker} 매수 - 가격: {price:,.2f}원, 수량: {quantity:.8f}")
            return True
            
        except Exception as e:
            logging.error(f"{ticker} 매수 실행 중 오류: {str(e)}")
            return False

    def execute_sell(self, ticker, price, current_time, reason):
        """매도 실행"""
        try:
            if ticker not in self.holdings:
                return False
                
            holding = self.holdings[ticker]
            quantity = holding['amount']
            buy_price = holding['buy_price']
            
            # 수익률 계산
            profit_rate = ((price - buy_price) / buy_price) * 100
            
            # 매도 금액 계산 (수수료 고려)
            sell_amount = quantity * price * (1 - Settings.Trading.COMMISSION_RATE)
            
            # 잔고 업데이트
            self.current_balance += sell_amount
            
            # 거래 기록
            trade_record = {
                'timestamp': current_time,
                'ticker': ticker,
                'action': 'sell',
                'price': price,
                'quantity': quantity,
                'profit_rate': profit_rate,
                'reason': reason
            }
            self.trades.append(trade_record)
            
            # 보유 코인에서 제거
            del self.holdings[ticker]
            
            logging.info(f"{ticker} 매도 - 가격: {price:,.2f}원, 수익률: {profit_rate:.2f}%, 사유: {reason}")
            return True
            
        except Exception as e:
            logging.error(f"{ticker} 매도 실행 중 오류: {str(e)}")
            return False

    def print_daily_summary(self, date):
        """일별 거래 요약 출력"""
        try:
            daily_trades = [t for t in self.trades if pd.to_datetime(t['timestamp']).date() == date.date()]
            
            if not daily_trades:
                logging.info(f"\n=== {date.date()} 거래 없음 ===")
                return
                
            buy_trades = [t for t in daily_trades if t['action'] == 'buy']
            sell_trades = [t for t in daily_trades if t['action'] == 'sell']
            
            # 수익 거래만 필터링
            profit_trades = [t for t in sell_trades if t.get('profit_rate', 0) > 0]
            
            total_profit = sum(t.get('profit_rate', 0) for t in sell_trades)
            avg_profit = total_profit / len(sell_trades) if sell_trades else 0
            
            logging.info(f"\n=== {date.date()} 거래 요약 ===")
            logging.info(f"총 거래 수: {len(daily_trades)}건")
            logging.info(f"매수: {len(buy_trades)}건")
            logging.info(f"매도: {len(sell_trades)}건")
            if sell_trades:
                logging.info(f"평균 수익률: {avg_profit:.2f}%")
                logging.info(f"수익 거래: {len(profit_trades)}건")
            logging.info(f"현재 잔고: {self.current_balance:,.0f}원")
            logging.info(f"일간 수익률: {((self.current_balance - self.initial_balance) / self.initial_balance) * 100:.2f}%")
            
        except Exception as e:
            logging.error(f"일별 요약 출력 중 오류: {str(e)}")

    def print_backtest_results(self):
        """백테스트 결과 출력"""
        try:
            total_trades = len(self.trades)
            sell_trades = [t for t in self.trades if t['action'] == 'sell']
            profit_trades = [t for t in sell_trades if t['profit_rate'] > 0]
            
            win_rate = (len(profit_trades) / len(sell_trades)) * 100 if sell_trades else 0
            total_profit = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
            
            logging.info("\n======= 백테스트 결과 =======")
            logging.info(f"테스트 기간: {self.start_date.date()} ~ {self.end_date.date()}")
            logging.info(f"초기 자본금: {self.initial_balance:,.0f}원")
            logging.info(f"최종 자본금: {self.current_balance:,.0f}원")
            logging.info(f"총 수익률: {total_profit:.2f}%")
            logging.info(f"총 거래 횟수: {total_trades}회")
            logging.info(f"승률: {win_rate:.2f}%")
            logging.info("============================\n")
            
        except Exception as e:
            logging.error(f"백테스트 결과 출력 중 오류: {str(e)}")

    def plot_results(self):
        """백테스트 결과 시각화"""
        try:
            # 일별 잔고 데이터 준비
            dates = sorted(self.daily_balance.keys())
            balances = [self.daily_balance[date]['balance'] for date in dates]
            profit_rates = [self.daily_balance[date]['profit_rate'] for date in dates]
            
            # 결과를 CSV 파일로 저장
            df = pd.DataFrame({
                'Date': dates,
                'Balance': balances,
                'Profit_Rate': profit_rates
            })
            df.to_csv('backtest_results.csv', index=False)
            logging.info("백테스트 결과가 'backtest_results.csv' 파일로 저장되었습니다.")
            
        except Exception as e:
            logging.error(f"결과 시각화 중 오류: {str(e)}")

    def save_results(self):
        """백테스트 결과 저장"""
        try:
            # 거래 기록 저장
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv('backtest_trades.csv', index=False)
            
            # 설정값 저장
            settings_dict = {
                'Trading': {k: v for k, v in vars(Settings.Trading).items() if not k.startswith('_')},
                'Technical': {k: v for k, v in vars(Settings.Technical).items() if not k.startswith('_')},
                'Market': {k: v for k, v in vars(Settings.Market).items() if not k.startswith('_')},
                'System': {k: v for k, v in vars(Settings.System).items() if not k.startswith('_')}
            }
            
            with open('backtest_settings.json', 'w') as f:
                json.dump(settings_dict, f, indent=4)
                
            logging.info("백테스트 결과가 저장되었습니다:")
            logging.info("- 거래 기록: backtest_trades.csv")
            logging.info("- 설정값: backtest_settings.json")
            
        except Exception as e:
            logging.error(f"결과 저장 중 오류: {str(e)}")

if __name__ == "__main__":
    main()