# RSI 역추세 전략 자동 백테스트 (다중 알트코인)
# 주요 구성: 전략 클래스 + 다중 티커 루프 + 성과 비교 분석

import os
import sys
import time
import logging
import pyupbit
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import multiprocessing as mp
from pathlib import Path        
import pickle
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import glob
import json
import colorama
from colorama import Fore, Back, Style
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import psutil
from tqdm import tqdm
from multiprocessing import Manager, Process, Queue

# 캐시 디렉토리 설정
CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)

# 로그 색상 및 이모지 정의
class LogColors:
    INFO = Fore.CYAN
    WARNING = Fore.YELLOW
    ERROR = Fore.RED
    SUCCESS = Fore.GREEN
    RESET = Style.RESET_ALL

class LogEmoji:
    INFO = "ℹ️"
    WARNING = "⚠️"
    ERROR = "❌"
    SUCCESS = "✅"
    LOADING = "⏳"
    DATA = "📊"
    CACHE = "💾"
    TRADE = "💰"
    TIME = "⏰"
    PROFIT = "💹"
    LOSS = "📉"
    ENTRY = "📈"
    EXIT = "📉"
    RISK = "⚠️"
    POSITION = "📊"
    MARKET = "🏛️"
    SIGNAL = "🔔"
    SUMMARY = "📊"

class ColoredFormatter(logging.Formatter):
    """컬러 로그 포맷터"""
    
    def format(self, record):
        # 로그 레벨에 따른 색상과 이모지 설정
        if record.levelno == logging.INFO:
            color = LogColors.INFO
            emoji = LogEmoji.INFO
        elif record.levelno == logging.WARNING:
            color = LogColors.WARNING
            emoji = LogEmoji.WARNING
        elif record.levelno == logging.ERROR:
            color = LogColors.ERROR
            emoji = LogEmoji.ERROR
        else:
            color = LogColors.RESET
            emoji = ""
            
        # 메시지에 색상과 이모지 추가
        record.msg = f"{color}{emoji} {record.msg}{LogColors.RESET}"
        return super().format(record)

# 로깅 설정
logger = logging.getLogger(__name__)

def setup_logging():
    """로깅 설정 초기화"""
    global logger
    
    # 기존 핸들러 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 로그 포맷 간소화
    log_format = '%(message)s'
    
    # 콘솔 핸들러만 사용
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter(log_format))
    
    # 루트 로거 설정
    logger = logging.getLogger(__name__)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    
    # 다른 로거들의 전파 방지
    logger.propagate = False

# 로깅 초기화 실행
setup_logging()

# 현재 디렉토리의 data_fetcher 모듈 임포트
current_dir = os.path.dirname(os.path.abspath(__file__))
data_fetcher_path = os.path.join(current_dir, 'data_fetcher.py')

if os.path.exists(data_fetcher_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('data_fetcher', data_fetcher_path)
    data_fetcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_fetcher)
    
    # data_fetcher에서 필요한 함수와 클래스 가져오기
    fetch_ohlcv = data_fetcher.fetch_ohlcv
    LogEmoji = data_fetcher.LogEmoji
    clean_old_cache = data_fetcher.clean_old_cache
else:
    raise FileNotFoundError(f"data_fetcher.py를 찾을 수 없습니다: {data_fetcher_path}")

warnings.filterwarnings('ignore')

# 전략 상수
DEFAULT_RSI_PERIOD = 14
DEFAULT_BB_PERIOD = 20
DEFAULT_BB_STD = 2.0
DEFAULT_TREND_PERIOD = 20
DEFAULT_VOL_PERIOD = 20
DEFAULT_POSITION_SIZE = 0.02  # 2%
DEFAULT_MAX_RISK_PER_TRADE = 0.02  # 2%
DEFAULT_STOP_LOSS_ATR_MULT = 2.0
DEFAULT_TAKE_PROFIT_ATR_MULT = 3.0
MIN_TRADE_AMOUNT = 5000  # 최소 거래금액 (원)

# 백테스팅 상수
DEFAULT_TEST_PERIOD_DAYS = 30
MAX_TEST_PERIOD_DAYS = 365
MIN_DATA_POINTS = 100
MAX_RETRIES = 5
RETRY_DELAY = 1.0

# 시장 상태 임계값
VOLATILITY_HIGH_THRESHOLD = 0.02  # 2%
TREND_STRENGTH_HIGH_THRESHOLD = 0.7
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# 캐시 설정
CACHE_MAX_AGE_DAYS = 7
CACHE_KEY_LENGTH = 16

# colorama 초기화
colorama.init()

def print_progress(current: int, total: int, prefix: str = '', suffix: str = '', decimals: int = 1, length: int = 50, fill: str = '█', print_end: str = "\n"):
    """
    진행률을 프로그레스 바 형태로 출력하는 함수
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end, flush=True)

def check_ticker_validity(ticker: str) -> bool:
    """티커의 유효성을 검사합니다."""
    try:
        # 현재가 조회로 유효성 체크
        current_price = pyupbit.get_current_price(ticker)
        return current_price is not None
    except Exception as e:
        logging.error(f"티커 유효성 검사 실패 ({ticker}): {str(e)}")
        return False

def safe_request_with_retry(func, *args, max_retries: int = 5, delay: float = 1.0, **kwargs):
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"{LogEmoji.ERROR} 시도 {attempt + 1}/{max_retries} 실패: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"{LogEmoji.TIME} {delay}초 후 재시도...")
                time.sleep(delay)
            else:
                logger.error(f"{LogEmoji.ERROR} 최대 재시도 횟수 초과")
                raise

def get_cache_key(ticker: str, start_date: str, end_date: str) -> str:
    """캐시 키 생성 (코인별 고유 캐시)"""
    try:
        # 날짜 형식 통일
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")
        
        # 코인별 고유 캐시 키 생성
        key = f"{ticker}_{start_dt}_{end_dt}"
        cache_key = hashlib.md5(key.encode()).hexdigest()[:16]
        
        logger.debug(f"{LogEmoji.CACHE} 캐시 키 생성: {ticker} -> {cache_key}")
        return cache_key
        
    except Exception as e:
        logger.error(f"캐시 키 생성 중 오류: {str(e)}")
        return hashlib.md5(f"{ticker}_{start_date}_{end_date}".encode()).hexdigest()[:16]

def print_data_progress(current: float, total: float, month: str, prefix: str = '', length: int = 40):
    # 진행률 바만 남기기 위해 모든 print 제거
    pass

def fetch_ohlcv(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    try:
        # logger.info(f"📊 {ticker} 데이터 로드 중...")
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        cache_files = []
        current_month = start_dt.replace(day=1)
        while current_month <= end_dt:
            cache_key = f"{ticker}_{current_month.strftime('%Y%m')}"
            cache_file = CACHE_DIR / f"{cache_key}.pkl"
            cache_files.append((current_month, cache_file))
            current_month = (current_month + timedelta(days=32)).replace(day=1)
        cached_data = []
        missing_months = []
        for month_start, cache_file in cache_files:
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                        if 'data' in cache_data and len(cache_data['data']) > 0:
                            cached_data.append(cache_data['data'])
                            continue
                except Exception as e:
                    # logger.warning(f"캐시 파일 로드 실패 ({cache_file.name})")
                    pass
            missing_months.append((month_start, cache_file))
        for month_start, cache_file in missing_months:
            month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            month_str = month_start.strftime('%Y-%m')
            month_data = []
            current_to = min(month_end, end_dt)
            est_total = int(((month_end - month_start).total_seconds() // (5*60)) + 1)
            total_count = 0
            while current_to >= month_start:
                for retry in range(5):
                    try:
                        df_part = pyupbit.get_ohlcv(ticker, interval="minute5", 
                                                   to=current_to.strftime('%Y-%m-%d %H:%M:%S'),
                                                   count=200)
                        if df_part is None or len(df_part) == 0:
                            if retry < 4:
                                time.sleep(1.0 * (2 ** retry))
                                continue
                            break
                        month_data.append(df_part)
                        current_to = df_part.index[0] - timedelta(minutes=5)
                        total_count += len(df_part)
                        time.sleep(0.3)
                        break
                    except Exception as e:
                        if retry < 4:
                            time.sleep(1.0 * (2 ** retry))
                        else:
                            break
            if month_data:
                df_month = pd.concat(month_data)
                df_month = df_month[~df_month.index.duplicated(keep='first')]
                df_month.sort_index(inplace=True)
                cache_data = {
                    'ticker': ticker,
                    'data': df_month,
                    'timestamp': datetime.now().timestamp(),
                    'month': month_str
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                cached_data.append(df_month)
        if not cached_data:
            # logger.error(f"❌ {ticker}: 데이터 수집 실패")
            return None
        df = pd.concat(cached_data)
        # 진단용 print 제거
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        df = df[start_date:end_date]
        # 진단용 print 제거
        if len(df) > 0:
            # logger.info(f"✅ {ticker} 데이터 로드 완료 ({len(df):,}개)")
            return df
        else:
            # logger.error(f"❌ {ticker}: 데이터가 비어있음")
            return None
    except Exception as e:
        # logger.error(f"❌ {ticker}: 데이터 조회 실패")
        return None

class MarketCondition(Enum):
    STRONG_UPTREND = "강한 상승추세"
    WEAK_UPTREND = "약한 상승추세"
    SIDEWAYS = "횡보장"
    WEAK_DOWNTREND = "약한 하락추세"
    STRONG_DOWNTREND = "강한 하락추세"
    HIGH_VOLATILITY = "고변동성"
    LOW_VOLATILITY = "저변동성"

@dataclass
class TradeMetrics:
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    market_condition: MarketCondition = MarketCondition.SIDEWAYS
    volatility: float = 0.0
    trade_count: int = 0

@dataclass
class TradingSummary:
    total_trades: int = 0
    successful_trades: int = 0
    total_profit_pct: float = 0.0
    current_progress: float = 0.0
    start_time: datetime = None
    last_update: datetime = None
    
    def update(self, profit_pct: float, success: bool):
        self.total_trades += 1
        if success:
            self.successful_trades += 1
        self.total_profit_pct += profit_pct
        
    @property
    def win_rate(self) -> float:
        return (self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
    @property
    def avg_profit(self) -> float:
        return self.total_profit_pct / self.total_trades if self.total_trades > 0 else 0

class RSIReversalStrategy:
    def __init__(self, ticker: str, is_backtest: bool = False, randomize: bool = True):
        self.ticker = ticker
        self.is_backtest = is_backtest
        
        # 코인별 최적화된 파라미터 설정
        if 'BTC' in ticker:
            self.rsi_period = random.randint(5, 8) if randomize else 6  # 더 짧게
            self.bb_period = random.randint(10, 15) if randomize else 12  # 더 짧게
            self.bb_std = random.uniform(1.5, 1.8) if randomize else 1.6  # 더 좁게
            self.volatility_threshold = 0.008  # 변동성 기준 낮춤
        elif 'ETH' in ticker:
            self.rsi_period = random.randint(4, 7) if randomize else 5
            self.bb_period = random.randint(8, 12) if randomize else 10
            self.bb_std = random.uniform(1.4, 1.7) if randomize else 1.5
            self.volatility_threshold = 0.01
        else:  # 알트코인 (현재 설정 유지)
            self.rsi_period = random.randint(3, 8) if randomize else 5
            self.bb_period = random.randint(8, 12) if randomize else 10
            self.bb_std = random.uniform(1.2, 1.8) if randomize else 1.5
            self.volatility_threshold = 0.025
        
        # 진입/청산 임계값 조정
        if 'BTC' in ticker or 'ETH' in ticker:
            self.rsi_entry_low = 42  # 더 공격적으로
            self.rsi_entry_high = 58
            self.trailing_stop_pct = 0.005  # 더 타이트하게
            self.profit_target_pct = 0.008  # 더 작은 익절
            self.max_hold_time = 8  # 더 짧은 보유시간
        else:
            self.rsi_entry_low = 40
            self.rsi_entry_high = 60
            self.trailing_stop_pct = 0.008
            self.profit_target_pct = 0.01
            self.max_hold_time = 12
        
        # 거래 기록 초기화
        self.trades = []
        self.metrics = TradeMetrics()
        self.summary = TradingSummary(start_time=datetime.now())
        self.last_progress_update = datetime.now()
        self.progress_update_interval = timedelta(seconds=1)
        
        # 현재 포지션 상태
        self.current_position = None
        self.entry_price = 0
        self.entry_time = None
        self.position_size = 0
        self.trailing_stop_price = 0
        
        # 성과 집계
        self.total_trades = 0
        self.successful_trades = 0
        self.total_profit_pct = 0.0
        self.df = None

    def analyze_market_condition(self, df: pd.DataFrame) -> Tuple[MarketCondition, bool]:
        try:
            # 단기/장기 변동성 (기간 단축)
            vol_short = df['close'].pct_change().rolling(6).std()  # 12 -> 6으로 단축
            vol_long = df['close'].pct_change().rolling(36).std()  # 72 -> 36으로 단축
            
            # 추세 강도 (기간 단축)
            ma3 = df['close'].rolling(3).mean()  # 5 -> 3으로 단축
            ma10 = df['close'].rolling(10).mean()  # 20 -> 10으로 단축
            trend_strength = (ma3.iloc[-1] - ma10.iloc[-1]) / ma10.iloc[-1]
            
            # 거래량 프로파일 (기간 단축)
            volume_sma = df['volume'].rolling(6).mean()  # 12 -> 6으로 단축
            volume_ratio = df['volume'].iloc[-1] / volume_sma.iloc[-1]
            
            # 시장 상태 판단 (기준 완화)
            if vol_short.iloc[-1] > vol_long.iloc[-1] * 2.0:  # 1.5 -> 2.0으로 완화
                return MarketCondition.HIGH_VOLATILITY, True  # False -> True로 변경
            elif vol_short.iloc[-1] < vol_long.iloc[-1] * 0.3:  # 0.5 -> 0.3으로 완화
                return MarketCondition.LOW_VOLATILITY, True
            elif abs(trend_strength) > 0.01:  # 0.02 -> 0.01로 완화
                if trend_strength > 0:
                    return MarketCondition.STRONG_UPTREND, True
                else:
                    return MarketCondition.STRONG_DOWNTREND, True
            else:
                return MarketCondition.SIDEWAYS, volume_ratio > 0.5  # 0.8 -> 0.5로 완화
                
        except Exception as e:
            logger.error(f"시장 상태 분석 중 오류: {str(e)}")
            return MarketCondition.SIDEWAYS, True  # False -> True로 변경

    def calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """RSI(Relative Strength Index) 계산"""
        try:
            # 가격 변화 계산
            delta = df['close'].diff()
            
            # 상승/하락 구분
            gain = (delta.where(delta > 0, 0))
            loss = (-delta.where(delta < 0, 0))
            
            # 평균 계산
            avg_gain = gain.rolling(window=self.rsi_period).mean()
            avg_loss = loss.rolling(window=self.rsi_period).mean()
            
            # RSI 계산
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"RSI 계산 중 오류: {str(e)}")
            return pd.Series(index=df.index)

    def calculate_bollinger_bands(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """볼린저 밴드 계산"""
        try:
            # 중심선 (20일 이동평균)
            middle = df['close'].rolling(window=self.bb_period).mean()
            
            # 표준편차
            std = df['close'].rolling(window=self.bb_period).std()
            
            # 상단/하단 밴드
            upper = middle + (std * self.bb_std)
            lower = middle - (std * self.bb_std)
            
            return lower, middle, upper
            
        except Exception as e:
            logger.error(f"볼린저 밴드 계산 중 오류: {str(e)}")
            return pd.Series(index=df.index), pd.Series(index=df.index), pd.Series(index=df.index)

    def calculate_dynamic_exits(self, df: pd.DataFrame, current_price: float) -> Tuple[float, float, float]:
        try:
            # 초단기 변동성 계산 (5분봉 기준)
            volatility_5min = df['close'].pct_change().rolling(5).std()
            volatility_1hour = df['close'].pct_change().rolling(12).std()
            
            # 변동성 비율에 따른 동적 조정
            vol_ratio = volatility_5min.iloc[-1] / volatility_1hour.iloc[-1]
            
            # 기본 스탑로스 설정
            base_stop = current_price * 0.005  # 0.5% 기본 스탑
            
            if vol_ratio > 1.5:  # 단기 변동성이 높을 때
                trailing_stop = base_stop * 1.5
            elif vol_ratio < 0.5:  # 단기 변동성이 낮을 때
                trailing_stop = base_stop * 0.8
            else:
                trailing_stop = base_stop
            
            # 최대 손실 제한
            max_stop = current_price * 0.015  # 최대 1.5% 손실
            trailing_stop = min(trailing_stop, max_stop)
            
            # 이익실현 목표 설정
            take_profit = current_price * (1 + (trailing_stop / current_price) * 2)
            
            return trailing_stop, take_profit, trailing_stop
            
        except Exception as e:
            logger.error(f"동적 청산가격 계산 중 오류: {str(e)}")
            return current_price * 0.005, current_price * 0.01, current_price * 0.005

    def _confirm_entry(self, df: pd.DataFrame, entry_type: str) -> bool:
        try:
            ma2 = df['close'].rolling(2).mean()
            ma3 = df['close'].rolling(3).mean()
            bb_std = df['close'].rolling(self.bb_period).std()
            bb_middle = df['close'].rolling(self.bb_period).mean()
            bb_upper = bb_middle + bb_std * self.bb_std
            bb_lower = bb_middle - bb_std * self.bb_std
            current_price = df['close'].iloc[-1]
            volatility = df['close'].pct_change().rolling(6).std().iloc[-1]
            if volatility < self.volatility_threshold * 0.7:  # 더 느슨하게
                return False
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(6).mean().iloc[-1]
            min_volume_ratio = 0.2  # 더 낮게
            if entry_type == "LONG":
                if 'BTC' in self.ticker or 'ETH' in self.ticker:
                    price_condition = current_price < bb_middle.iloc[-1] * 1.01  # 더 느슨하게
                else:
                    price_condition = current_price < bb_lower.iloc[-1] * 1.01
                ma_condition = ma2.iloc[-1] > ma2.iloc[-2]
                volume_condition = volume_ratio > min_volume_ratio
                return price_condition and (ma_condition or volume_condition)
            else:  # SHORT
                if 'BTC' in self.ticker or 'ETH' in self.ticker:
                    price_condition = current_price > bb_middle.iloc[-1] * 0.99  # 더 느슨하게
                else:
                    price_condition = current_price > bb_upper.iloc[-1] * 0.99
                ma_condition = ma2.iloc[-1] < ma2.iloc[-2]
                volume_condition = volume_ratio > min_volume_ratio
                return price_condition and (ma_condition or volume_condition)
        except Exception as e:
            logger.error(f"진입 확인 중 오류: {str(e)}")
            return False

    def evaluate_entry(self, df: pd.DataFrame) -> Tuple[bool, str]:
        try:
            rsi = self.calculate_rsi(df)
            current_rsi = rsi.iloc[-1]
            price_change = df['close'].pct_change(2)
            current_change = price_change.iloc[-1]
            # RSI 기준 완화, 가격 변화율 기준 완화
            if current_rsi < 60:
                entry_type = "LONG"
                if current_change < 0.002:
                    if self._confirm_entry(df, "LONG"):
                        return True, entry_type
            elif current_rsi > 40:
                entry_type = "SHORT"
                if current_change > -0.002:
                    if self._confirm_entry(df, "SHORT"):
                        return True, entry_type
            return False, ""
        except Exception as e:
            logger.error(f"진입 조건 평가 중 오류: {str(e)}")
            return False, ""

    def calculate_position_size(self, current_price: float, stop_loss_pct: float) -> float:
        try:
            # 기본 설정
            base_position = 0.1  # 기본 10% 포지션
            
            # 승률에 따른 조정
            win_rate = self.summary.win_rate / 100
            position_mult = min(1.5, max(0.5, win_rate * 2))
            
            # 수익률에 따른 조정
            profit_factor = min(1.2, max(0.8, 1 + (self.summary.total_profit_pct / 10)))
            
            # 최종 포지션 크기 계산
            position_size = base_position * position_mult * profit_factor
            
            # 안전장치
            max_position = 0.2  # 최대 20%
            min_position = 0.05  # 최소 5%
            
            return min(max_position, max(min_position, position_size))
            
        except Exception as e:
            logger.error(f"포지션 크기 계산 중 오류: {str(e)}")
            return 0.05

    def evaluate_exit(self, df: pd.DataFrame, position: str, entry_price: float, hold_time: float) -> Tuple[bool, str]:
        """개선된 청산 조건 평가"""
        try:
            current_price = df['close'].iloc[-1]
            
            # 동적 청산가격 계산
            stop_loss, take_profit, trailing_stop = self.calculate_dynamic_exits(df, entry_price)
            
            # 트레일링 스탑 업데이트
            if position == 'long':
                if self.trailing_stop_price == 0:
                    self.trailing_stop_price = entry_price - trailing_stop
                else:
                    new_stop = current_price - trailing_stop
                    self.trailing_stop_price = max(self.trailing_stop_price, new_stop)
            else:  # short position
                if self.trailing_stop_price == 0:
                    self.trailing_stop_price = entry_price + trailing_stop
                else:
                    new_stop = current_price + trailing_stop
                    self.trailing_stop_price = min(self.trailing_stop_price, new_stop)
            
            # 청산 조건 확인
            profit_pct = ((current_price - entry_price) / entry_price * 100)
            if position == 'short':
                profit_pct = -profit_pct
            
            # 트레일링 스탑 히트
            if position == 'long' and current_price < self.trailing_stop_price:
                return True, "트레일링 스탑"
            if position == 'short' and current_price > self.trailing_stop_price:
                return True, "트레일링 스탑"
            
            # 이익실현
            if profit_pct >= self.profit_target_pct:
                return True, "이익실현"
            
            # 최대 보유시간
            if hold_time >= self.max_hold_time:
                return True, "보유시간 초과"
            
            # 시장 상태 변화
            market_condition, is_tradeable = self.analyze_market_condition(df)
            if not is_tradeable:
                return True, "시장상태 변화"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"청산 조건 평가 중 오류: {str(e)}")
            return True, "오류"

    def process_single_window(self, window: pd.DataFrame) -> List[Dict]:
        """단일 데이터 윈도우에 대한 거래 처리"""
        try:
            if len(window) < self.bb_period:
                return []
            
            current_price = float(window['close'].iloc[-1])
            current_time = window.index[-1]
            
            # 현재 포지션이 없는 경우 진입 조건 확인
            if self.current_position is None:
                should_enter, entry_type = self.evaluate_entry(window)
                if should_enter:
                    # 포지션 크기 계산
                    stop_loss, _, _ = self.calculate_dynamic_exits(window, current_price)
                    stop_loss_pct = (stop_loss / current_price - 1) if entry_type == "LONG" else (1 - stop_loss / current_price)
                    self.position_size = self.calculate_position_size(current_price, abs(stop_loss_pct))
                    
                    if self.position_size > 0:
                        self.current_position = entry_type.lower()
                        self.entry_price = current_price
                        self.entry_time = current_time
                        self.trailing_stop_price = 0
                        
                        # 거래 기록 (로그 출력 없이)
                        self.log_trade_details('ENTRY', current_price,
                            position=self.current_position,
                            position_size=self.position_size,
                            reason="진입 조건 충족")
            
            # 현재 포지션이 있는 경우 청산 조건 확인
            else:
                hold_time = (current_time - self.entry_time).total_seconds() / 3600
                should_exit, exit_reason = self.evaluate_exit(window, self.current_position, 
                                                            self.entry_price, hold_time)
                
                if should_exit:
                    # 수익률 계산
                    profit_pct = ((current_price - self.entry_price) / self.entry_price * 100)
                    if self.current_position == 'short':
                        profit_pct = -profit_pct
                    
                    # 거래 기록 (로그 출력 없이)
                    self.log_trade_details('EXIT', current_price,
                        position=self.current_position,
                        profit_pct=profit_pct,
                        reason=exit_reason)
                    
                    # 포지션 초기화
                    self.current_position = None
                    self.entry_price = 0
                    self.entry_time = None
                    self.position_size = 0
                    self.trailing_stop_price = 0
            
            return self.trades
            
        except Exception as e:
            logger.error(f"거래 처리 중 오류: {str(e)}")
            return []

    def update_progress(self, current_idx: int, total_size: int):
        """진행 상황 업데이트 및 출력"""
        now = datetime.now()
        if (now - self.last_progress_update) >= self.progress_update_interval:
            self.summary.current_progress = (current_idx / total_size) * 100
            self.summary.last_update = now
            # 진행률 출력
            elapsed_time = (now - self.summary.start_time).total_seconds()
            estimated_total = elapsed_time / (self.summary.current_progress / 100) if self.summary.current_progress > 0 else 0
            remaining_time = max(0, estimated_total - elapsed_time)
            # 시간 형식 변환
            minutes = int(remaining_time // 60)
            seconds = int(remaining_time % 60)
            time_str = f"{minutes}분 {seconds}초" if minutes > 0 else f"{seconds}초"
            # 진행률 표시 (여러 코인 동시 출력 위해 줄바꿈)
            print_progress(
                current_idx, total_size,
                prefix=self.ticker,
                suffix=f"거래:{self.summary.total_trades}건 수익:{self.summary.total_profit_pct:+.2f}% 남은시간:{time_str}",
                length=30,
                print_end='\n'
            )
            self.last_progress_update = now

    def log_trade_details(self, trade_type: str, price: float, **kwargs):
        """거래 상세 정보 기록"""
        try:
            trade_info = {
                'type': trade_type,
                'price': price,
                'timestamp': datetime.now(),
                **kwargs
            }
            
            self.trades.append(trade_info)
            
            # 거래 요약 업데이트 (로그 출력 없이)
            if trade_type == 'EXIT':
                profit_pct = kwargs.get('profit_pct', 0)
                success = profit_pct > 0
                self.summary.update(profit_pct, success)
            
        except Exception as e:
            logger.error(f"거래 기록 중 오류: {str(e)}")

    def log_performance_summary(self):
        """전략 성과 요약 출력"""
        try:
            if not self.trades:
                logger.warning(f"{LogEmoji.WARNING} {self.ticker}: 거래 기록 없음")
                return
            
            # 시장 상태 분석
            market_condition, _ = self.analyze_market_condition(self.df)
            
            # 성과 지표 계산
            total_trades = len([t for t in self.trades if t['type'] == 'EXIT'])
            if total_trades == 0:
                logger.warning(f"{LogEmoji.WARNING} {self.ticker}: 완료된 거래 없음")
                return
                
            successful_trades = len([t for t in self.trades if t['type'] == 'EXIT' and t.get('profit_pct', 0) > 0])
            win_rate = (successful_trades / total_trades) * 100
            
            profits = [t.get('profit_pct', 0) for t in self.trades if t['type'] == 'EXIT']
            total_profit = sum(profits)
            avg_profit = total_profit / total_trades if total_trades > 0 else 0
            
            # 결과 출력
            logger.info(f"\n{LogEmoji.SUMMARY} {self.ticker} 전략 성과 요약:")
            logger.info(f"시장 상태: {market_condition.value}")
            logger.info(f"총 거래: {total_trades}건")
            logger.info(f"승률: {win_rate:.1f}%")
            logger.info(f"총 수익: {total_profit:.2f}%")
            logger.info(f"평균 수익: {avg_profit:.2f}%")
            
            # 등급 평가
            grade = self._evaluate_performance_grade(self.metrics)
            logger.info(f"종합 등급: {grade}\n")
            
        except Exception as e:
            logger.error(f"성과 요약 출력 중 오류: {str(e)}")

    def _evaluate_performance_grade(self, metrics: TradeMetrics) -> str:
        """성과 등급 평가"""
        try:
            # 샤프 비율 기준
            if metrics.sharpe_ratio >= 2.0:
                return "S"
            elif metrics.sharpe_ratio >= 1.5:
                return "A"
            elif metrics.sharpe_ratio >= 1.0:
                return "B"
            elif metrics.sharpe_ratio >= 0.5:
                return "C"
            else:
                return "D"
                
        except Exception as e:
            logger.error(f"등급 평가 중 오류: {str(e)}")
            return "F"

    def calculate_metrics(self) -> TradeMetrics:
        """전략 성과 지표 계산"""
        try:
            metrics = TradeMetrics()
            
            if not self.trades:
                return metrics
            
            # 기본 지표 계산
            exit_trades = [t for t in self.trades if t['type'] == 'EXIT']
            total_trades = len(exit_trades)
            
            if total_trades == 0:
                return metrics
            
            # 승률
            successful_trades = len([t for t in exit_trades if t.get('profit_pct', 0) > 0])
            metrics.win_rate = (successful_trades / total_trades) * 100
            
            # 수익성 지표
            profits = [t.get('profit_pct', 0) for t in exit_trades]
            gains = [p for p in profits if p > 0]
            losses = [p for p in profits if p <= 0]
            
            metrics.avg_profit = sum(gains) / len(gains) if gains else 0
            metrics.avg_loss = sum(losses) / len(losses) if losses else 0
            
            # 프로핏 팩터
            total_gains = sum(gains)
            total_losses = abs(sum(losses))
            metrics.profit_factor = total_gains / total_losses if total_losses != 0 else float('inf')
            
            # 최대 낙폭
            cumulative = np.cumsum(profits)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = running_max - cumulative
            metrics.max_drawdown = np.max(drawdowns)
            
            # 샤프 비율
            returns = pd.Series(profits)
            excess_returns = returns - 0.02  # 2% 무위험 수익률 가정
            metrics.sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if len(returns) > 1 else 0
            
            # 시장 상태
            metrics.market_condition, _ = self.analyze_market_condition(self.df)
            
            # 변동성
            metrics.volatility = self.df['close'].pct_change().std() * np.sqrt(252)
            metrics.trade_count = total_trades
            
            return metrics
            
        except Exception as e:
            logger.error(f"지표 계산 중 오류: {str(e)}")
            return TradeMetrics()

def compare_strategies(strategies: List[RSIReversalStrategy]) -> pd.DataFrame:
    """전략 비교 결과 출력 (간소화된 버전)"""
    try:
        results = []
        for strategy in strategies:
            metrics = strategy.calculate_metrics()
            results.append({
                '티커': strategy.ticker,
                '시장상태': metrics.market_condition.value,
                '승률': f"{metrics.win_rate:.1f}%",
                '총수익': f"{strategy.summary.total_profit_pct:.2f}%",
                '등급': strategy._evaluate_performance_grade(metrics)
            })
        
        # 결과를 데이터프레임으로 변환
        df_results = pd.DataFrame(results)
        
        # 결과 출력
        print("\n" + "="*70)
        print("📊 전략 성과 비교")
        print("-"*70)
        print(df_results.to_string(index=False))
        print("="*70 + "\n")
        
        return df_results
        
    except Exception as e:
        logger.error(f"전략 비교 중 오류: {str(e)}")
        return pd.DataFrame()

# 캐시 관리 개선
def clean_old_cache(cache_dir: Path = CACHE_DIR, max_age_days: int = CACHE_MAX_AGE_DAYS):
    """오래된 캐시 파일 정리 (월별 캐시 지원)"""
    try:
        now = datetime.now()
        cleaned = 0
        for cache_file in cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    
                    # 캐시 데이터의 월 확인
                    if 'month' in cache_data:
                        cache_month = datetime.strptime(cache_data['month'], '%Y-%m')
                        age_months = (now.year - cache_month.year) * 12 + now.month - cache_month.month
                        
                        # 6개월 이상 된 캐시는 삭제
                        if age_months > 6:
                            cache_file.unlink()
                            cleaned += 1
                            logger.info(f"캐시 삭제: {cache_data['month']} ({cache_file.name})")
                    
                    # 이전 형식의 캐시는 삭제
                    elif 'timestamp' in cache_data:
                        data_age = (now - datetime.fromtimestamp(cache_data['timestamp'])).days
                        if data_age > max_age_days:
                            cache_file.unlink()
                            cleaned += 1
                    
            except Exception as e:
                logger.warning(f"캐시 파일 처리 중 오류 ({cache_file.name}): {str(e)}")
                cache_file.unlink()  # 손상된 캐시 파일 삭제
                cleaned += 1
        
        if cleaned > 0:
            logger.info(f"오래된 캐시 파일 {cleaned}개 정리 완료")
            
    except Exception as e:
        logger.error(f"캐시 정리 중 오류: {str(e)}")

def performance_log_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.debug(f"{LogEmoji.TIME} {func.__name__} 실행 시간: {execution_time:.2f}초")
        return result
    return wrapper

def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"{LogEmoji.DATA} 메모리 사용량: {memory_info.rss / 1024 / 1024:.2f} MB")

class BacktestResult:
    def __init__(self, ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.total_trades = 0
        self.win_rate = 0.0
        self.total_profit = 0.0
        self.avg_profit = 0.0
        self.max_drawdown = 0.0
        self.market_condition = ""
        self.timestamp = datetime.now()

    def to_dict(self):
        return {
            'ticker': self.ticker,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'total_profit': self.total_profit,
            'avg_profit': self.avg_profit,
            'max_drawdown': self.max_drawdown,
            'market_condition': self.market_condition,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }

def save_backtest_results(results: List[BacktestResult]):
    """백테스트 결과를 JSON 파일로 저장"""
    try:
        # 결과 저장 디렉토리 생성
        results_dir = Path(__file__).parent / 'backtest_results'
        results_dir.mkdir(exist_ok=True)
        
        # 결과 파일명 생성 (타임스탬프 포함)
        filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = results_dir / filename
        
        # 결과를 JSON으로 변환
        results_data = [result.to_dict() for result in results]
        
        # JSON 파일로 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"{LogEmoji.SUCCESS} 백테스트 결과 저장 완료: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"{LogEmoji.ERROR} 결과 저장 중 오류: {str(e)}")
        return None

def analyze_backtest_history():
    """과거 백테스트 결과 분석"""
    try:
        results_dir = Path(__file__).parent / 'backtest_results'
        if not results_dir.exists():
            logger.warning(f"{LogEmoji.WARNING} 백테스트 기록이 없습니다")
            return
            
        # 모든 결과 파일 로드
        all_results = []
        for file in results_dir.glob('*.json'):
            with open(file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                all_results.extend(results)
        
        if not all_results:
            logger.warning(f"{LogEmoji.WARNING} 분석할 결과가 없습니다")
            return
            
        # 결과 분석
        df = pd.DataFrame(all_results)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 코인별 성과 분석
        logger.info(f"\n{LogEmoji.SUMMARY} 코인별 평균 성과:")
        coin_stats = df.groupby('ticker').agg({
            'total_trades': 'mean',
            'win_rate': 'mean',
            'total_profit': 'mean',
            'avg_profit': 'mean'
        }).round(2)
        
        print("\n" + "="*70)
        print("코인별 평균 성과")
        print("-"*70)
        print(coin_stats)
        print("="*70)
        
        # 시장상태별 성과 분석
        logger.info(f"\n{LogEmoji.SUMMARY} 시장상태별 평균 성과:")
        market_stats = df.groupby('market_condition').agg({
            'total_trades': 'mean',
            'win_rate': 'mean',
            'total_profit': 'mean',
            'avg_profit': 'mean'
        }).round(2)
        
        print("\n" + "="*70)
        print("시장상태별 평균 성과")
        print("-"*70)
        print(market_stats)
        print("="*70)
        
        # 성과 추세 분석
        logger.info(f"\n{LogEmoji.SUMMARY} 전략 성과 추세:")
        df['date'] = df['timestamp'].dt.date
        trend = df.groupby('date').agg({
            'total_profit': 'mean',
            'win_rate': 'mean'
        }).round(2)
        
        print("\n" + "="*70)
        print("일자별 평균 성과")
        print("-"*70)
        print(trend.tail())
        print("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"{LogEmoji.ERROR} 결과 분석 중 오류: {str(e)}")

def run_strategy_for_ticker(ticker, start_date_str, end_date_str, progress_queue):
    try:
        df = fetch_ohlcv(ticker, start_date_str, end_date_str)
        # 진단용 print 제거
        if df is None or len(df) < MIN_DATA_POINTS:
            progress_queue.put((ticker, 'done', None))
            return None
        strategy = RSIReversalStrategy(ticker, is_backtest=True, randomize=True)
        strategy.df = df
        total_len = len(df)
        for i in range(total_len):
            window = df.iloc[max(0, i-strategy.bb_period):i+1]
            if len(window) >= strategy.bb_period:
                strategy.process_single_window(window)
            progress_queue.put((ticker, 'progress', {
                'current': i+1,
                'total': total_len,
                'trades': strategy.summary.total_trades,
                'profit': strategy.summary.total_profit_pct
            }))
        progress_queue.put((ticker, 'done', None))
        return strategy
    except Exception as e:
        progress_queue.put((ticker, 'done', None))
        return None

if __name__ == "__main__":
    try:
        # 테스트할 티커 목록
        tickers = [
            "KRW-BTC",  # 비트코인
            "KRW-ETH",  # 이더리움
            "KRW-XRP",  # 리플
            "KRW-BCH",  # 비트코인캐시(폴리곤 대신)
            "KRW-ADA",  # 에이다
            "KRW-DOGE", # 도지코인
        ]
        # 백테스트 기간 랜덤 설정
        test_periods = [30, 60, 90, 180]  # 테스트 기간 옵션 (일)
        selected_period = random.choice(test_periods)  # 랜덤하게 하나 선택
        # 3년 내의 임의의 기간 선택
        current_date = datetime.now()
        three_years_ago = current_date - timedelta(days=365*3)
        # 선택된 기간을 고려하여 랜덤 시작일 결정
        max_start_date = current_date - timedelta(days=selected_period)
        random_days = random.randint(0, (max_start_date - three_years_ago).days)
        start_date = three_years_ago + timedelta(days=random_days)
        end_date = start_date + timedelta(days=selected_period)
        # 날짜를 문자열로 변환
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        # ====== 기간을 눈에 띄게 출력 ======
        print("\n" + "="*60)
        print(f"[백테스트 기간] {selected_period}일  ({start_date_str} ~ {end_date_str})")
        print("="*60 + "\n")
        logger.info(f"{LogEmoji.INFO} 백테스트 시작")
        logger.info(f"대상 티커: {', '.join(tickers)}")
        logger.info(f"테스트 기간: {selected_period}일 ({start_date_str} ~ {end_date_str})")

        # tqdm 멀티바 준비
        manager = Manager()
        progress_queue = manager.Queue()
        bars = {}
        bar_data = {}
        for idx, ticker in enumerate(tickers):
            bars[ticker] = tqdm(total=1, desc=ticker, position=idx, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
            bar_data[ticker] = {'total': 1}

        # 워커 프로세스 실행
        processes = []
        for ticker in tickers:
            p = Process(target=run_strategy_for_ticker, args=(ticker, start_date_str, end_date_str, progress_queue))
            p.start()
            processes.append(p)

        finished = set()
        strategies = []
        while len(finished) < len(tickers):
            try:
                ticker, msg_type, data = progress_queue.get(timeout=1)
                if msg_type == 'progress':
                    if data['total'] != bar_data[ticker]['total']:
                        bars[ticker].reset(total=data['total'])
                        bar_data[ticker]['total'] = data['total']
                    bars[ticker].n = data['current']
                    bars[ticker].set_postfix({
                        '거래': data['trades'],
                        '수익': f"{data['profit']:+.2f}%"
                    })
                    bars[ticker].refresh()
                elif msg_type == 'done':
                    finished.add(ticker)
                    bars[ticker].n = bars[ticker].total
                    bars[ticker].refresh()
            except Exception:
                pass
        for p in processes:
            p.join()
        for bar in bars.values():
            bar.close()

        # 전략 비교 및 랭킹
        if strategies:
            compare_strategies(strategies)
        # 백테스트 결과 저장
        backtest_results = []
        for strategy in strategies:
            result = BacktestResult(
                ticker=strategy.ticker,
                start_date=start_date_str,
                end_date=end_date_str
            )
            result.total_trades = strategy.summary.total_trades
            result.win_rate = strategy.summary.win_rate
            result.total_profit = strategy.summary.total_profit_pct
            result.avg_profit = strategy.summary.avg_profit
            result.max_drawdown = strategy.metrics.max_drawdown
            result.market_condition = strategy.metrics.market_condition.value
            backtest_results.append(result)
        # 결과 저장 및 분석
        save_backtest_results(backtest_results)
        analyze_backtest_history()
        logger.info(f"\n{LogEmoji.SUCCESS} 백테스트 완료")
    except Exception as e:
        logger.error(f"{LogEmoji.ERROR} 프로그램 실행 중 오류: {str(e)}")