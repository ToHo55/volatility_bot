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
        BASE_TRADE_AMOUNT = 100000      # 기본 거래 금액 (원)
        MIN_TRADE_AMOUNT = 50000        # 최소 거래 금액 (원)
        COMMISSION_RATE = 0.005         # 거래 수수료 비율
        MAX_HOLDINGS = 5                # 보유 가능한 최대 코인 수
        CHECK_INTERVAL = 5              # 사이클 간격 (초)

        # 매매 전략 설정
        CONFIDENCE_THRESHOLD = 65       # 종합 신뢰도가 65% 이상이면 매수
        TAKE_PROFIT_PERCENT = 2.0       # 익절 목표: 2% 이익 발생 시
        STOP_LOSS_PERCENT = 1.0         # 손절 기준: 1% 손실 발생 시
        FORCE_SELL_MINUTES = 60         # 매수 후 60분 경과 시 강제 청산

    class Technical:
        # RSI 설정
        RSI_PERIOD = 14
        RSI_OVERSOLD = 35
        RSI_WEIGHT = 0.4               # RSI 전략 가중치

        # 볼린저 밴드 설정
        BB_WINDOW = 20
        BB_STD_DEV = 2
        BB_WEIGHT = 0.4                # BB 전략 가중치

        # 변동성 돌파 설정
        VOLATILITY_K = 0.5             # 변동성 계수
        VOLATILITY_WEIGHT = 0.4        # 변동성 돌파 전략 가중치

        # 이동평균선 설정
        MA_SHORT = 5                   # 단기 이동평균선
        MA_LONG = 20                   # 장기 이동평균선
        MA_WEIGHT = 0.2                # 이동평균선 전략 가중치

    class Market:
        # 거래량 필터링
        MIN_TRADE_VOLUME = 50000        # 최소 거래량 조건 (5분봉 기준)
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
        try:
            rsi = Indicators.calculate_rsi(df, Settings.Technical.RSI_PERIOD)
            _, _, bb_pos = Indicators.calculate_bollinger_bands(
                df, 
                Settings.Technical.BB_WINDOW,
                Settings.Technical.BB_STD_DEV
            )
            
            if rsi is None or bb_pos is None:
                return None
                
            return {
                'rsi': rsi,
                'bb_position': bb_pos,
                'strategy': 'RSI_BB'
            }
            
        except Exception as e:
            logger.error(f"RSI/BB 분석 중 오류: {str(e)}")
            return None
            
    def calculate_confidence(self, analysis_result):
        if not analysis_result:
            return 0
            
        rsi = analysis_result['rsi']
        bb_pos = analysis_result['bb_position']
        
        # RSI 신뢰도 계산
        if rsi <= 30:
            rsi_conf = 90 + (30 - rsi)
        elif rsi <= 40:
            rsi_conf = 70 + (40 - rsi) * 2
        elif rsi <= 50:
            rsi_conf = 50 + (50 - rsi) * 2
        elif rsi <= 60:
            rsi_conf = 30 + (60 - rsi) * 2
        elif rsi <= 70:
            rsi_conf = 10 + (70 - rsi) * 2
        else:
            rsi_conf = max(0, 100 - rsi)
            
        # BB 위치 신뢰도 계산
        if bb_pos <= 20:
            bb_conf = 80 + (20 - bb_pos)
        elif bb_pos <= 40:
            bb_conf = 60 + (40 - bb_pos)
        elif bb_pos <= 60:
            bb_conf = 40 + (60 - bb_pos) / 2
        elif bb_pos <= 80:
            bb_conf = 20 + (80 - bb_pos) / 2
        else:
            bb_conf = max(0, 100 - bb_pos)
            
        return (rsi_conf * 0.5 + bb_conf * 0.5)

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

class StrategyComposer:
    """전략 조합기"""
    def __init__(self):
        self.strategies = {
            'RSI_BB': (RSIBBStrategy(), 0.4),      # 40% 비중
            'VOLATILITY': (VolatilityBreakoutStrategy(), 0.4),  # 40% 비중
            'MA': (MAStrategy(), 0.2)              # 20% 비중
        }
        
    def analyze(self, ticker, df, current_price):
        try:
            results = {}
            confidences = []
            
            for name, (strategy, weight) in self.strategies.items():
                result = strategy.analyze(ticker, df, current_price)
                if result:
                    results[name] = result
                    confidence = strategy.calculate_confidence(result)
                    confidences.append(confidence * weight)
                    
            if not confidences:
                return None
                
            # 종합 신뢰도 계산
            total_confidence = sum(confidences)
            
            return {
                'total_confidence': total_confidence,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"전략 분석 중 오류: {str(e)}")
            return None

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
    """API 호출 래퍼 함수"""
    while True:
        try:
            result = func(*args, **kwargs)
            time.sleep(Settings.System.API_REQUEST_INTERVAL)
            return result
        except Exception as e:
            logger.error(f"API 호출 오류 ({func.__name__}): {e}. 백오프 중...")
            time.sleep(5)

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
    global cycle_count, emergency_stop
    
    # 초기화
    trading_logic = TradingLogic()
    load_trading_state()
    
    last_volume_update = datetime.now(KST)
    filtered_markets = []
    
    while True:
        try:
            if emergency_stop:
                logger.info("비상정지 신호 감지. 프로그램을 종료합니다.")
                break

            cycle_count += 1
            current_cycle_counts['error'] = 0
            logger.info(f"===== 사이클 {cycle_count} 시작 =====")
            
            # 매 사이클마다 시장 상태 업데이트
            markets = load_markets()
            if not markets:
                logger.warning("거래 가능한 마켓이 없습니다.")
                time.sleep(Settings.Trading.CHECK_INTERVAL)
                continue
                
            # 거래량 기준 필터링 (30분마다)
            if not filtered_markets or (datetime.now(KST) - last_volume_update).total_seconds() >= Settings.Market.VOLUME_UPDATE_INTERVAL * 60:
                filtered_markets = filter_by_volume(markets)
                last_volume_update = datetime.now(KST)
                logger.info("거래량 필터링 업데이트 완료")
                
            if not filtered_markets:
                logger.warning("거래량 조건을 만족하는 코인이 없습니다.")
                time.sleep(Settings.Trading.CHECK_INTERVAL)
                continue
            
            # 전체 시장 상태 분석
            market_weight = analyze_market_state(filtered_markets)
            
            # 보유 코인 수 확인
            logger.info(f"현재 보유 코인 ({len(trading_logic.held_coins)}/{Settings.Trading.MAX_HOLDINGS}개): {list(trading_logic.held_coins.keys())}")
            
            # 보유 코인 점검
            sold_coins = []  # 이번 사이클에서 매도된 코인
            for ticker in list(trading_logic.held_coins.keys()):
                try:
                    current_price = pyupbit.get_current_price(ticker)
                    if current_price is None:
                        continue
                        
                    buy_data = trading_logic.held_coins[ticker]
                    action, reason = trading_logic.execute_trade_decision(
                        ticker, 
                        current_price,
                        buy_data["buy_time"],
                        buy_data["buy_price"]
                    )
                    
                    if action == "SELL":
                        if trading_logic.execute_sell(ticker, current_price, reason):
                            sold_coins.append(ticker)
                            if reason == "take_profit":
                                current_cycle_counts['tp'] += 1
                                cumulative_counts['tp'] += 1
                            elif reason == "stop_loss":
                                current_cycle_counts['sl_normal'] += 1
                                cumulative_counts['sl_normal'] += 1
                            elif reason == "force_sell":
                                current_cycle_counts['sl_forced'] += 1
                                cumulative_counts['sl_forced'] += 1
                            else:
                                current_cycle_counts['sl_other'] += 1
                                cumulative_counts['sl_other'] += 1
                        
                except Exception as e:
                    current_cycle_counts['error'] += 1
                    logger.error(f"{ticker} 보유 코인 점검 중 오류: {str(e)}")
            
            # 새로운 매수 기회 탐색
            available_balance = upbit.get_balance("KRW")
            if available_balance < Settings.Trading.MIN_TRADE_AMOUNT:
                logger.info(f"사용 가능한 KRW 잔고 부족: {available_balance:,.0f}원")
            else:
                logger.info(f"매수 가능 KRW 잔고: {available_balance:,.0f}원")
                
                # 모든 코인을 매수 대상으로 검토
                buy_candidates = [ticker for ticker in filtered_markets 
                                if ticker not in sold_coins]
                
                try:
                    current_prices = pyupbit.get_current_price(buy_candidates)
                    if not current_prices:
                        logger.error("현재가 조회 실패")
                        continue
                        
                    if not isinstance(current_prices, dict):
                        current_prices = {buy_candidates[0]: current_prices}
                        
                    logger.debug(f"현재가 조회 결과: {len(current_prices)}개 코인")
                    
                    for ticker in buy_candidates[:10]:
                        try:
                            if ticker not in current_prices:
                                continue
                                
                            current_price = current_prices[ticker]
                            if not current_price or current_price == 0:
                                continue
                                
                            try:
                                current_price = float(current_price)
                            except (TypeError, ValueError):
                                continue
                                
                            action, info = trading_logic.execute_trade_decision(ticker, current_price)
                            
                            if action == "BUY":
                                if trading_logic.execute_buy(ticker, current_price, market_weight, info):
                                    current_cycle_counts['buy'] += 1
                                    cumulative_counts['buy'] += 1
                                    logger.info(f"{ticker} 매수 완료")
                                
                        except Exception as e:
                            current_cycle_counts['error'] += 1
                            logger.error(f"{ticker} 매수 기회 탐색 중 오류: {str(e)}")
                        
                except Exception as e:
                    logger.error(f"현재가 일괄 조회 중 오류: {str(e)}")
            
            # 캐시 정리 (매 10 사이클마다)
            if cycle_count % 10 == 0:
                ohlcv_cache.clear()
                indicator_cache.clear()
                logger.info("캐시 정리 완료")
            
            # 거래 내역 저장 (매 10 사이클마다)
            if cycle_count % 10 == 0:
                save_trade_history()
            
            log_cycle_summary(cycle_count, trading_logic)
            logger.info(f"===== 사이클 {cycle_count} 종료, {Settings.Trading.CHECK_INTERVAL}초 대기 =====")
            time.sleep(Settings.Trading.CHECK_INTERVAL)
            
            # 주기적 데이터 관리
            trading_logic.data_manager.calculate_statistics(cycle_count)
            trading_logic.data_manager.monitor_performance(cycle_count)
            
            if cycle_count % 1440 == 0:  # 하루에 한 번
                trading_logic.data_manager.manage_data_files()
                
        except KeyboardInterrupt:
            logger.warning("키보드 인터럽트가 감지되었습니다. 비상정지를 실행합니다.")
            emergency_stop_handler(None, None)
            break
        except Exception as e:
            logger.error(f"메인 루프 실행 중 오류 발생: {str(e)}")
            time.sleep(Settings.Trading.CHECK_INTERVAL)

def load_trading_state():
    """거래 상태 로드"""
    global trade_history
    try:
        # 거래 기록 불러오기
        if os.path.exists("trade_history.json"):
            with open("trade_history.json", "r", encoding="utf-8") as f:
                trade_history = json.load(f)
        
        logger.info("거래 상태 로드 완료")
    except Exception as e:
        logger.error(f"거래 상태 로드 중 오류 발생: {str(e)}")

def save_trade_history():
    """거래 내역 저장"""
    try:
        with open("trade_history.json", "w", encoding="utf-8") as f:
            json.dump(trade_history, f, ensure_ascii=False, indent=4)
        logger.debug("거래 내역 저장 완료")
    except Exception as e:
        logger.error(f"거래 내역 저장 중 오류: {str(e)}")

def log_cycle_summary(cycle, trading_logic_instance):
    """사이클 요약 정보를 로깅"""
    try:
        # 현재 보유 코인 상태
        current_holdings = len(trading_logic_instance.held_coins)
        total_invested = sum(coin_data['amount'] * coin_data['buy_price'] 
                           for coin_data in trading_logic_instance.held_coins.values())
        
        logger.info("\n===== 사이클 요약 =====")
        logger.info(f"사이클 번호: {cycle}")
        logger.info(f"현재 보유 코인: {current_holdings}개")
        logger.info(f"총 투자금액: {total_invested:,.0f}원")
        
        logger.info("\n[현재 사이클]")
        logger.info(f"- 매수: {current_cycle_counts['buy']}건")
        logger.info(f"- 익절: {current_cycle_counts['tp']}건")
        logger.info(f"- 손절 상세:")
        logger.info(f"  * 일반 손절: {current_cycle_counts['sl_normal']}건")
        logger.info(f"  * 강제 청산: {current_cycle_counts['sl_forced']}건")
        logger.info(f"  * 기타 손절: {current_cycle_counts['sl_other']}건")
        logger.info(f"- 에러: {current_cycle_counts['error']}건")
        
        logger.info("\n[누적 현황]")
        logger.info(f"- 총 매수: {cumulative_counts['buy']}건")
        logger.info(f"- 총 익절: {cumulative_counts['tp']}건")
        logger.info(f"- 손절 누적:")
        logger.info(f"  * 일반 손절: {cumulative_counts['sl_normal']}건")
        logger.info(f"  * 강제 청산: {cumulative_counts['sl_forced']}건")
        logger.info(f"  * 기타 손절: {cumulative_counts['sl_other']}건")
        logger.info(f"- 총 에러: {cumulative_counts['error']}건")
        
        # 수익률 정보
        if cycle % 10 == 0:  # 10 사이클마다 수익률 계산
            try:
                with open(trading_logic_instance.data_manager.log_file, 'r', encoding='utf-8') as f:
                    df = pd.read_csv(f)
                    if not df.empty:
                        total_profit = df[df['action'] == 'sell']['profit_loss'].sum()
                        avg_profit = df[df['action'] == 'sell']['profit_loss'].mean()
                        win_rate = (df[df['action'] == 'sell']['profit_loss'] > 0).mean() * 100
                        
                        logger.info("\n[수익률 분석]")
                        logger.info(f"- 총 수익률: {total_profit:.2f}%")
                        logger.info(f"- 평균 수익률: {avg_profit:.2f}%")
                        logger.info(f"- 승률: {win_rate:.2f}%")
            except Exception as e:
                logger.error(f"수익률 계산 중 오류: {str(e)}")
        
        logger.info("=====================\n")

    except Exception as e:
        logger.error(f"사이클 요약 로깅 중 오류: {str(e)}")
    finally:
        # 현재 사이클 카운터 초기화
        for key in current_cycle_counts:
            current_cycle_counts[key] = 0

def emergency_stop_handler(signum, frame):
    """비상정지 핸들러"""
    global emergency_stop
    emergency_stop = True
    logger.warning("비상정지 신호가 감지되었습니다. 모든 코인을 매도합니다...")
    sell_all_coins()

class Backtesting:
    def __init__(self, start_date, end_date, initial_balance=10000000):
        """
        백테스팅 클래스 초기화
        :param start_date: 시작일 (YYYY-MM-DD)
        :param end_date: 종료일 (YYYY-MM-DD)
        :param initial_balance: 초기 자본금 (원)
        """
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trading_logic = TradingLogic()
        self.holdings = {}  # 보유 코인 정보
        self.trade_history = []  # 거래 기록
        self.daily_balance = []  # 일별 잔고 기록

    def run_backtest(self, tickers=None):
        """백테스팅 실행"""
        if tickers is None:
            # KRW 마켓의 모든 코인 (스테이블 코인 제외)
            tickers = [ticker for ticker in pyupbit.get_tickers(fiat="KRW") 
                      if ticker not in Settings.Market.STABLE_COINS]

        logger.info(f"백테스팅 시작: {self.start_date.date()} ~ {self.end_date.date()}")
        logger.info(f"초기 자본금: {self.initial_balance:,}원")
        logger.info(f"대상 코인 수: {len(tickers)}개")

        current_date = self.start_date
        while current_date <= self.end_date:
            self.simulate_day(current_date, tickers)
            self.record_daily_balance(current_date)
            current_date += timedelta(days=1)

        self.print_backtest_results()
        self.plot_results()
        self.save_results()

    def simulate_day(self, date, tickers):
        """하루 동안의 거래 시뮬레이션"""
        try:
            # 5분 단위로 시뮬레이션
            for minute in range(0, 24*60, 5):
                current_time = date + timedelta(minutes=minute)
                
                # 보유 코인 점검
                self.check_holdings(current_time)
                
                # 새로운 매수 기회 탐색
                if len(self.holdings) < Settings.Trading.MAX_HOLDINGS:
                    self.find_buying_opportunities(current_time, tickers)

        except Exception as e:
            logger.error(f"시뮬레이션 중 오류 발생: {str(e)}")

    def check_holdings(self, current_time):
        """보유 코인 점검"""
        for ticker, data in list(self.holdings.items()):
            try:
                df = pyupbit.get_ohlcv(ticker, interval="minute5", 
                                     to=current_time.strftime("%Y%m%d %H%M%S"),
                                     count=100)
                if df is None or df.empty:
                    continue

                current_price = df['close'].iloc[-1]
                buy_price = data['buy_price']
                profit_rate = (current_price - buy_price) / buy_price * 100

                # 매도 조건 확인
                should_sell = False
                sell_reason = ""

                # 익절
                if profit_rate >= Settings.Trading.TAKE_PROFIT_PERCENT:
                    should_sell = True
                    sell_reason = "take_profit"
                # 손절
                elif profit_rate <= -Settings.Trading.STOP_LOSS_PERCENT:
                    should_sell = True
                    sell_reason = "stop_loss"
                # 강제 청산
                elif (current_time - data['buy_time']).seconds > Settings.Trading.FORCE_SELL_MINUTES * 60:
                    should_sell = True
                    sell_reason = "force_sell"

                if should_sell:
                    self.execute_sell(ticker, current_price, current_time, sell_reason)

            except Exception as e:
                logger.error(f"{ticker} 보유 코인 점검 중 오류: {str(e)}")

    def find_buying_opportunities(self, current_time, tickers):
        """매수 기회 탐색"""
        for ticker in tickers:
            if ticker in self.holdings:
                continue

            try:
                df = pyupbit.get_ohlcv(ticker, interval="minute5",
                                     to=current_time.strftime("%Y%m%d %H%M%S"),
                                     count=100)
                if df is None or df.empty:
                    continue

                current_price = df['close'].iloc[-1]
                
                # 거래량 필터링
                if df['volume'].mean() < Settings.Market.MIN_TRADE_VOLUME:
                    continue

                # 매수 신호 확인
                action, confidence = self.trading_logic.should_buy(ticker, current_price)
                
                if action and confidence >= Settings.Trading.CONFIDENCE_THRESHOLD:
                    self.execute_buy(ticker, current_price, current_time, confidence)

            except Exception as e:
                logger.error(f"{ticker} 매수 기회 탐색 중 오류: {str(e)}")

    def execute_buy(self, ticker, price, time, confidence):
        """매수 실행"""
        available_balance = self.balance
        if available_balance < Settings.Trading.MIN_TRADE_AMOUNT:
            return

        trade_amount = min(Settings.Trading.BASE_TRADE_AMOUNT,
                         available_balance * 0.9995)  # 수수료 고려
        
        quantity = trade_amount / price
        total_cost = trade_amount * (1 + Settings.Trading.COMMISSION_RATE)

        self.balance -= total_cost
        self.holdings[ticker] = {
            'amount': quantity,
            'buy_price': price,
            'buy_time': time,
            'confidence': confidence
        }

        self.trade_history.append({
            'time': time,
            'ticker': ticker,
            'action': 'buy',
            'price': price,
            'quantity': quantity,
            'confidence': confidence,
            'balance': self.balance
        })

    def execute_sell(self, ticker, price, time, reason):
        """매도 실행"""
        holding = self.holdings[ticker]
        quantity = holding['amount']
        buy_price = holding['buy_price']
        
        total_return = price * quantity * (1 - Settings.Trading.COMMISSION_RATE)
        profit_loss = ((price - buy_price) / buy_price) * 100

        self.balance += total_return
        del self.holdings[ticker]

        self.trade_history.append({
            'time': time,
            'ticker': ticker,
            'action': 'sell',
            'price': price,
            'quantity': quantity,
            'reason': reason,
            'profit_loss': profit_loss,
            'balance': self.balance
        })

    def record_daily_balance(self, date):
        """일별 잔고 기록"""
        total_value = self.balance
        for ticker, data in self.holdings.items():
            try:
                df = pyupbit.get_ohlcv(ticker, interval="day", to=date.strftime("%Y%m%d"), count=1)
                if df is not None and not df.empty:
                    current_price = df['close'].iloc[-1]
                    total_value += current_price * data['amount']
            except Exception as e:
                logger.error(f"{ticker} 현재가 조회 중 오류: {str(e)}")

        self.daily_balance.append({
            'date': date.date(),
            'total_value': total_value
        })

    def print_backtest_results(self):
        """백테스팅 결과 출력"""
        if not self.trade_history or not self.daily_balance:
            logger.warning("거래 기록이 없습니다.")
            return

        total_trades = len([t for t in self.trade_history if t['action'] == 'sell'])
        winning_trades = len([t for t in self.trade_history 
                            if t['action'] == 'sell' and t['profit_loss'] > 0])
        
        initial_balance = self.initial_balance
        final_balance = self.daily_balance[-1]['total_value']
        total_return = ((final_balance - initial_balance) / initial_balance) * 100
        
        logger.info("\n===== 백테스팅 결과 =====")
        logger.info(f"테스트 기간: {self.start_date.date()} ~ {self.end_date.date()}")
        logger.info(f"초기 자본금: {initial_balance:,.0f}원")
        logger.info(f"최종 자본금: {final_balance:,.0f}원")
        logger.info(f"총 수익률: {total_return:.2f}%")
        logger.info(f"총 거래 횟수: {total_trades}회")
        logger.info(f"승률: {(winning_trades/total_trades*100):.2f}%")
        
        # 월별 수익률 계산
        monthly_returns = self.calculate_monthly_returns()
        logger.info("\n[월별 수익률]")
        for month, return_rate in monthly_returns.items():
            logger.info(f"{month}: {return_rate:.2f}%")

    def calculate_monthly_returns(self):
        """월별 수익률 계산"""
        monthly_returns = {}
        
        for i in range(len(self.daily_balance)-1):
            current_date = self.daily_balance[i]['date']
            month_key = current_date.strftime("%Y-%m")
            
            if month_key not in monthly_returns:
                monthly_returns[month_key] = []
            
            daily_return = ((self.daily_balance[i+1]['total_value'] - 
                           self.daily_balance[i]['total_value']) / 
                          self.daily_balance[i]['total_value'] * 100)
            monthly_returns[month_key].append(daily_return)
        
        # 월별 평균 수익률 계산
        return {month: sum(returns) for month, returns in monthly_returns.items()}

    def plot_results(self):
        """결과 시각화"""
        try:
            import matplotlib.pyplot as plt
            
            # 자본금 추이
            dates = [b['date'] for b in self.daily_balance]
            values = [b['total_value'] for b in self.daily_balance]
            
            plt.figure(figsize=(12, 6))
            plt.plot(dates, values, label='Portfolio Value')
            plt.title('Backtest Results')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value (KRW)')
            plt.grid(True)
            plt.legend()
            
            # 파일로 저장
            plt.savefig('backtest_results.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"결과 시각화 중 오류 발생: {str(e)}")

    def save_results(self):
        """백테스팅 결과 저장"""
        try:
            results = {
                'settings': {
                    'start_date': self.start_date.strftime("%Y-%m-%d"),
                    'end_date': self.end_date.strftime("%Y-%m-%d"),
                    'initial_balance': self.initial_balance
                },
                'trade_history': self.trade_history,
                'daily_balance': self.daily_balance
            }
            
            with open('backtest_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4, default=str)
                
            logger.info("백테스팅 결과가 'backtest_results.json'에 저장되었습니다.")
            
        except Exception as e:
            logger.error(f"결과 저장 중 오류 발생: {str(e)}")

def run_backtest(start_date, end_date, initial_balance=10000000):
    """백테스팅 실행 함수"""
    backtester = Backtesting(start_date, end_date, initial_balance)
    backtester.run_backtest()
    return backtester

if __name__ == "__main__":
    try:
        # 로깅 시스템 초기화
        setup_logging()
        logger.info("자동매매 프로그램을 시작합니다.")
        
        # 설정값 출력
        print_settings()
        
        # API 키 유효성 검사
        if not UPBIT_ACCESS or not UPBIT_SECRET:
            raise ValueError("API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
            
        # Upbit 객체 초기화 전에 키 유효성 검사
        try:
            upbit = pyupbit.Upbit(UPBIT_ACCESS, UPBIT_SECRET)
            balance = upbit.get_balance("KRW")
            if balance is None:
                raise ValueError("잔고 조회 실패. API 키를 확인해주세요.")
            logger.info(f"초기 KRW 잔고: {balance:,.0f}원")
        except Exception as e:
            raise ValueError(f"Upbit 초기화 실패: {str(e)}")
            
        # 메인 루프 시작
        main()
    except KeyboardInterrupt:
        logger.warning("프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {str(e)}")
    finally:
        if emergency_stop:
            sell_all_coins()
        logger.info("프로그램이 종료되었습니다.") 