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
from collections import Counter

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
FEE_RATE = 0.0005  # 업비트 현물 기준 0.05%
SLIPPAGE = 0.0002  # 메이저 코인 기준 0.02% (보수적이면 0.0005 유지)

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
    STRONG_UPTREND_HIGH_VOL = "상승+고변동성"
    STRONG_DOWNTREND = "강한 하락추세"
    STRONG_DOWNTREND_HIGH_VOL = "하락+고변동성"
    SIDEWAYS = "횡보장"
    SIDEWAYS_HIGH_VOL = "횡보+고변동성"
    WEAK_UPTREND = "약한 상승추세"
    WEAK_DOWNTREND = "약한 하락추세"
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

# ====== 진입 후 수익률 곡세 기반 진입 금지 market_condition 자동 추출 ======
bad_conditions = set()
try:
    import pandas as pd
    import os
    curve_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'entry_curve_data.csv')
    if os.path.exists(curve_path):
        curve = pd.read_csv(curve_path, names=['market_condition', 'offset', 'ret'], encoding='utf-8')
        for mc in curve['market_condition'].unique():
            means = curve[curve['market_condition'] == mc].groupby('offset')['ret'].mean()
            if all(means < 0):
                bad_conditions.add(mc)
        print('진입 금지 market_condition:', bad_conditions)
except Exception as e:
    print('진입 금지 market_condition 자동 추출 오류:', e)

class RSIReversalStrategy:
    def __init__(self, ticker: str, is_backtest: bool = False, randomize: bool = True):
        self.ticker = ticker
        self.is_backtest = is_backtest
        
        # 코인별 최적화된 파라미터 설정
        if ticker == "KRW-XRP":
            # XRP: 보수적
            self.rsi_period = random.randint(6, 10) if randomize else 8
            self.bb_period = random.randint(12, 18) if randomize else 15
            self.bb_std = random.uniform(1.6, 2.0) if randomize else 1.8
            self.volatility_threshold = 0.03
            self.min_volume_ratio = 0.1
        elif ticker in ["KRW-DOGE", "KRW-BCH", "KRW-ADA"]:
            # DOGE/BCH/ADA: 공격적
            self.rsi_period = random.randint(3, 6) if randomize else 4
            self.bb_period = random.randint(6, 10) if randomize else 8
            self.bb_std = random.uniform(1.1, 1.4) if randomize else 1.2
            self.volatility_threshold = 0.015
            self.min_volume_ratio = 0.02
        else:
            # 기타(혹시 대비)
            self.rsi_period = random.randint(4, 8) if randomize else 6
            self.bb_period = random.randint(8, 12) if randomize else 10
            self.bb_std = random.uniform(1.3, 1.7) if randomize else 1.5
            self.volatility_threshold = 0.025
            self.min_volume_ratio = 0.05
        
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
        self.bad_conditions = bad_conditions

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
            
            # 값 파일로 저장 (절대경로)
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'market_debug.log')
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.now().isoformat()},{self.ticker if hasattr(self,'ticker') else ''},{trend_strength:.6f},{vol_short.iloc[-1]:.6f},{vol_long.iloc[-1]:.6f},{volume_ratio:.6f}\n")
            except Exception as log_e:
                pass
            
            # ======= 복합 시장 상태 분류 =======
            HIGH_VOL = 0.0063
            UP = 0.0030
            DOWN = -0.0031
            if vol_short.iloc[-1] > HIGH_VOL:
                if trend_strength > UP:
                    return MarketCondition.STRONG_UPTREND_HIGH_VOL, True
                elif trend_strength < DOWN:
                    return MarketCondition.STRONG_DOWNTREND_HIGH_VOL, True
                else:
                    return MarketCondition.SIDEWAYS_HIGH_VOL, True
            else:
                if trend_strength > UP:
                    return MarketCondition.STRONG_UPTREND, True
                elif trend_strength < DOWN:
                    return MarketCondition.STRONG_DOWNTREND, True
                else:
                    return MarketCondition.SIDEWAYS, volume_ratio > 2.84

            # 실전 수익 기반 분류 추가
            if len(df) > 20:
                recent_returns = df['close'].pct_change(20).iloc[-1]
                if recent_returns > 0.002:
                    return MarketCondition.STRONG_UPTREND, True
                elif recent_returns < -0.002:
                    return MarketCondition.STRONG_DOWNTREND, True
        except Exception as e:
            logger.error(f"시장 상태 분석 중 오류: {str(e)}")
            return MarketCondition.SIDEWAYS, True

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
            # 트레일링 스탑 가변화: 변동성이 높을수록 더 여유롭게, 낮을수록 더 타이트하게
            if vol_ratio > 2.0:
                trailing_stop = base_stop * 2.0
            elif vol_ratio > 1.5:
                trailing_stop = base_stop * 1.5
            elif vol_ratio < 0.5:
                trailing_stop = base_stop * 0.5
            elif vol_ratio < 0.8:
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
            market_condition, _ = self.analyze_market_condition(df)
            # 진입 금지 market_condition 자동 적용
            if hasattr(self, 'bad_conditions') and market_condition.value in self.bad_conditions:
                return False
            # 시장상태별 전략 완전 분리/고도화
            if market_condition == MarketCondition.STRONG_UPTREND_HIGH_VOL:
                # 상승+고변동성: 추세추종, 신호 완화, 포지션 크게
                # 볼린저밴드 상단 돌파 + 거래량 급증 + RSI 60 이상 등
                bb_middle = df['close'].rolling(self.bb_period).mean()
                bb_upper = bb_middle + df['close'].rolling(self.bb_period).std() * self.bb_std
                current_price = df['close'].iloc[-1]
                volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(6).mean().iloc[-1]
                rsi = self.calculate_rsi(df)
                current_rsi = rsi.iloc[-1] if not rsi.isnull().all() else 50
                # 신호 완화: 볼린저밴드 상단 돌파 or 거래량 급증 or RSI 60 이상
                if (current_price > bb_upper.iloc[-1]*0.99) or (volume_ratio > 2.0) or (current_rsi > 60):
                    return True
                return False
            elif market_condition == MarketCondition.SIDEWAYS:
                # 횡보장: mean-reversion, 신호 엄격, 포지션 작게
                bb_middle = df['close'].rolling(self.bb_period).mean()
                bb_lower = bb_middle - df['close'].rolling(self.bb_period).std() * self.bb_std
                current_price = df['close'].iloc[-1]
                rsi = self.calculate_rsi(df)
                current_rsi = rsi.iloc[-1] if not rsi.isnull().all() else 50
                # 신호 엄격: 볼린저밴드 하단 터치 + RSI 35 이하만 진입
                if entry_type == "LONG" and (current_price < bb_lower.iloc[-1]*1.01) and (current_rsi < 35):
                    return True
                return False
            elif market_condition == MarketCondition.STRONG_DOWNTREND_HIGH_VOL:
                # 하락+고변동성: 진입 금지
                return False
            else:
                # 기타: 기존 로직
                return True
        except Exception as e:
            logger.error(f"진입 확인 중 오류: {str(e)}")
            return False

    def calculate_position_size(self, current_price: float, stop_loss_pct: float) -> float:
        try:
            market_condition, _ = self.analyze_market_condition(self.df)
            # 시장상태별 포지션 크기 완전 분리
            if market_condition == MarketCondition.STRONG_UPTREND_HIGH_VOL:
                return 0.2  # 최대치
            elif market_condition == MarketCondition.SIDEWAYS:
                return 0.05  # 최소치
            elif market_condition == MarketCondition.STRONG_DOWNTREND_HIGH_VOL:
                return 0.0  # 진입 금지
            else:
                return 0.1
        except Exception as e:
            logger.error(f"포지션 크기 계산 중 오류: {str(e)}")
            return 0.05

    def evaluate_exit(self, df: pd.DataFrame, position: str, entry_price: float, hold_time: float) -> Tuple[bool, str]:
        try:
            current_price = df['close'].iloc[-1]
            profit_pct = ((current_price - entry_price) / entry_price * 100)
            if position == 'short':
                profit_pct = -profit_pct
            market_condition, _ = self.analyze_market_condition(df)
            # 스마트 익절: +1.5% 이상이면 무조건 청산
            if profit_pct >= 1.5:
                return True, "스마트익절"
            # 시장상태별 익절/손절폭
            if market_condition == MarketCondition.STRONG_UPTREND_HIGH_VOL:
                if profit_pct >= 5.0:
                    return True, "강한상승_익절"
                if profit_pct <= -2.0:
                    return True, "강한상승_손절"
            elif market_condition == MarketCondition.SIDEWAYS:
                if profit_pct >= 3.0:
                    return True, "횡보_익절"
                if profit_pct <= -1.0:
                    return True, "횡보_손절"
            # 트레일링 스탑 강화: 진입 후 +1.0% 이상 갔다가 -0.2% 이탈 시 청산
            if not hasattr(self, 'max_profit_since_entry'):
                self.max_profit_since_entry = profit_pct
            self.max_profit_since_entry = max(self.max_profit_since_entry, profit_pct)
            if self.max_profit_since_entry >= 1.0 and profit_pct < (self.max_profit_since_entry - 1.2):
                return True, "트레일링스탑_강화"
            # 기타 기존 로직
            return False, ""
        except Exception as e:
            logger.error(f"청산 조건 평가 중 오류: {str(e)}")
            return True, "오류"

    def evaluate_entry(self, df: pd.DataFrame) -> Tuple[bool, str]:
        try:
            market_condition, _ = self.analyze_market_condition(df)
        # 코인 필터링: 성과 나쁜 코인, 변동성 낮은 코인 제외
        # if self.ticker in ["KRW-BCH", "KRW-ADA"]:
        #     return False, ""
        # if df['close'].pct_change().rolling(20).std().iloc[-1] < 0.005:
        #     return False, ""
        # 횡보장 대기조건: 추가 상승 확인 없으면 진입 보류
            if market_condition == MarketCondition.SIDEWAYS:
                prev_close = df['close'].iloc[-2]
                curr_close = df['close'].iloc[-1]
                if (curr_close - prev_close) / prev_close < 0.003:
                    return False, ""
            rsi = self.calculate_rsi(df)
            current_rsi = rsi.iloc[-1] if not rsi.isnull().all() else 50
            # RSI 30 이하 + 3봉 연속 하락
            if current_rsi < 30:
                if all(df['close'].diff().iloc[-i] < 0 for i in range(1, 4)):
                    return self._confirm_entry(df, "LONG"), "LONG"
                else:
                    return False, ""
            # 기존 신호(추세추종 등)
            price_change = df['close'].pct_change(2)
            current_change = price_change.iloc[-1] if not price_change.isnull().all() else 0
            bb_middle = df['close'].rolling(self.bb_period).mean()
            bb_upper = bb_middle + df['close'].rolling(self.bb_period).std() * self.bb_std
            bb_lower = bb_middle - df['close'].rolling(self.bb_period).std() * self.bb_std
            current_price = df['close'].iloc[-1]
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(6).mean().iloc[-1]
            recent_exits = [t for t in self.trades[-10:] if t['type'] == 'EXIT']
            recent_losses = [t for t in recent_exits if t.get('profit_pct', 0) < 0]
            if len(recent_losses) >= 3 and all(t.get('profit_pct', 0) < 0 for t in recent_exits[-3:]):
                return False, ""
            if market_condition == MarketCondition.STRONG_UPTREND_HIGH_VOL:
                if (current_rsi > 50 and current_price > bb_upper.iloc[-1] and volume_ratio > 1.2):
                    return self._confirm_entry(df, "LONG"), "LONG"
            elif market_condition == MarketCondition.STRONG_DOWNTREND_HIGH_VOL:
                if (current_rsi < 50 and current_price < bb_lower.iloc[-1] and volume_ratio > 1.2):
                    return self._confirm_entry(df, "SHORT"), "SHORT"
            else:
                if current_rsi < 40 and current_price < bb_lower.iloc[-1]*1.01:
                    return self._confirm_entry(df, "LONG"), "LONG"
                elif current_rsi > 60 and current_price > bb_upper.iloc[-1]*0.99:
                    return self._confirm_entry(df, "SHORT"), "SHORT"
            return False, ""
        except Exception as e:
            logger.error(f"진입 조건 평가 중 오류: {str(e)}")
            return False, ""

    def process_single_window(self, window: pd.DataFrame) -> List[Dict]:
        """단일 데이터 윈도우에 대한 거래 처리"""
        try:
            if len(window) < self.bb_period:
                return []
            current_price = float(window['close'].iloc[-1])
            current_time = window.index[-1]
            market_condition, _ = self.analyze_market_condition(window)
            # 복합 상태별 진입 허용 조건 분기
            if market_condition in [MarketCondition.SIDEWAYS, MarketCondition.SIDEWAYS_HIGH_VOL]:
                return self.trades
            # DOGE, BCH는 HIGH_VOLATILITY도 진입 허용 (복합 상태도 포함)
            if self.current_position is None:
                if self.ticker in ["KRW-DOGE", "KRW-BCH"]:
                    allowed_conditions = [MarketCondition.STRONG_UPTREND, MarketCondition.STRONG_DOWNTREND, MarketCondition.WEAK_UPTREND, MarketCondition.WEAK_DOWNTREND, MarketCondition.HIGH_VOLATILITY, MarketCondition.STRONG_UPTREND_HIGH_VOL, MarketCondition.STRONG_DOWNTREND_HIGH_VOL]
                else:
                    allowed_conditions = [MarketCondition.STRONG_UPTREND, MarketCondition.STRONG_DOWNTREND, MarketCondition.WEAK_UPTREND, MarketCondition.WEAK_DOWNTREND, MarketCondition.STRONG_UPTREND_HIGH_VOL, MarketCondition.STRONG_DOWNTREND_HIGH_VOL]
                if market_condition not in allowed_conditions:
                    return self.trades  # 진입 자체를 하지 않음
            orig_profit_target = self.profit_target_pct
            orig_trailing_stop = self.trailing_stop_pct
            if market_condition in [MarketCondition.SIDEWAYS, MarketCondition.SIDEWAYS_HIGH_VOL]:
                self.profit_target_pct = 0.003
                self.trailing_stop_pct = 0.002
            if market_condition in [MarketCondition.STRONG_UPTREND_HIGH_VOL, MarketCondition.STRONG_DOWNTREND_HIGH_VOL]:
                self.profit_target_pct = 0.005
                self.trailing_stop_pct = 0.003
            # 현재 포지션이 없는 경우 진입 조건 확인
            if self.current_position is None:
                should_enter, entry_type = self.evaluate_entry(window)
                if should_enter:
                    stop_loss, _, _ = self.calculate_dynamic_exits(window, current_price)
                    stop_loss_pct = (stop_loss / current_price - 1) if entry_type == "LONG" else (1 - stop_loss / current_price)
                    self.position_size = self.calculate_position_size(current_price, abs(stop_loss_pct))
                    if self.position_size > 0:
                        self.current_position = entry_type.lower()
                        self.entry_price = current_price * (1 + FEE_RATE + SLIPPAGE) if entry_type == "LONG" else current_price * (1 - FEE_RATE - SLIPPAGE)
                        self.entry_time = current_time
                        self.trailing_stop_price = 0
                        self.log_trade_details('ENTRY', current_price,
                            position=self.current_position,
                            position_size=self.position_size,
                            reason="진입 조건 충족",
                            market_condition=market_condition.value)
            # 현재 포지션이 있는 경우 청산 조건 확인
            else:
                hold_time = (current_time - self.entry_time).total_seconds() / 3600
                should_exit, exit_reason = self.evaluate_exit(window, self.current_position, 
                                                            self.entry_price, hold_time)
                if should_exit:
                    if self.current_position == 'long':
                        exit_price = current_price * (1 - FEE_RATE - SLIPPAGE)
                        profit_pct = ((exit_price - self.entry_price) / self.entry_price * 100)
                    else:  # short
                        exit_price = current_price * (1 + FEE_RATE + SLIPPAGE)
                        profit_pct = ((self.entry_price - exit_price) / self.entry_price * 100)
                    self.log_trade_details('EXIT', current_price,
                        position=self.current_position,
                        profit_pct=profit_pct,
                        reason=exit_reason,
                        market_condition=market_condition.value)
                    self.current_position = None
                    self.entry_price = 0
                    self.entry_time = None
                    self.position_size = 0
                    self.trailing_stop_price = 0
            self.profit_target_pct = orig_profit_target
            self.trailing_stop_pct = orig_trailing_stop
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
        try:
            trade_info = {
                'type': trade_type,
                'price': price,
                'timestamp': datetime.now(),
                **kwargs
            }
            self.trades.append(trade_info)
            if trade_type == 'EXIT':
                profit_pct = kwargs.get('profit_pct', 0)
                success = profit_pct > 0
                self.summary.update(profit_pct, success)
            # ENTRY/EXIT 시점의 market_condition 로그 추가 (7개 필드)
            market_condition = kwargs.get('market_condition', '')
            profit_pct = kwargs.get('profit_pct', '')
            reason = kwargs.get('reason', '')
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'market_debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()},{self.ticker},{trade_type},{price},{profit_pct},{reason},{market_condition}\n")
            # ENTRY 시점에 진입 후 10,20,30틱 뒤 수익률 저장
            if trade_type == 'ENTRY' and hasattr(self, 'df'):
                entry_idx = self.df[self.df['close'] == price].index[-1]
                idx = self.df.index.get_loc(entry_idx)
                for offset in [10, 20, 30]:
                    if idx + offset < len(self.df):
                        future_price = self.df['close'].iloc[idx + offset]
                        ret = (future_price - price) / price
                        with open('entry_curve_data.csv', 'a', encoding='utf-8') as ef:
                            ef.write(f'{market_condition},{offset},{ret}\n')
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
            # ===== 복합 상태 분포 리포트 추가 =====
            if self.df is not None:
                from collections import Counter
                all_market_conditions = []
                for i in range(len(self.df)):
                    window = self.df.iloc[max(0, i-self.bb_period):i+1]
                    if len(window) >= self.bb_period:
                        mc, _ = self.analyze_market_condition(window)
                        all_market_conditions.append(mc.value)
                if all_market_conditions:
                    logger.info(f"시장상태 분포: {dict(Counter(all_market_conditions))}")
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
        # 전략 객체를 파일로 저장
        with open(f'strategy_result_{ticker}.pkl', 'wb') as f:
            pickle.dump(strategy, f)
        return strategy
    except Exception as e:
        progress_queue.put((ticker, 'done', None))
        return None

if __name__ == "__main__":
    try:
        NUM_REPEAT = 1  # 반복 횟수
        all_results = []
        for repeat_idx in range(NUM_REPEAT):
            print(f"\n{'='*60}\n[반복 {repeat_idx+1}/{NUM_REPEAT}] 백테스트 시작\n{'='*60}")
            # 테스트할 티커 목록
            tickers = [
                "KRW-XRP",  # 리플
                "KRW-BCH",  # 비트코인캐시
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
            # 전체 기간의 시장상태 판별(최빈값)
            all_market_conditions = []
            for ticker in tickers:
                try:
                    df = fetch_ohlcv(ticker, start_date_str, end_date_str)
                    if df is not None and len(df) > 0:
                        strategy_tmp = RSIReversalStrategy(ticker, is_backtest=True, randomize=True)
                        for i in range(len(df)):
                            window = df.iloc[max(0, i-strategy_tmp.bb_period):i+1]
                            if len(window) >= strategy_tmp.bb_period:
                                market_condition, _ = strategy_tmp.analyze_market_condition(window)
                                all_market_conditions.append(market_condition.value)
                except Exception:
                    pass
            if all_market_conditions:
                most_common_market = Counter(all_market_conditions).most_common(1)[0][0]
                market_str = f" | 시장상태: {most_common_market}"
                # 시장상태 분포 출력
                print("[시장상태 분포]", dict(Counter(all_market_conditions)))
            else:
                market_str = ""
            print("\n" + "="*60)
            print(f"[백테스트 기간] {selected_period}일  ({start_date_str} ~ {end_date_str}){market_str}")
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
            strategies = []
            for ticker in tickers:
                p = Process(target=run_strategy_for_ticker, args=(ticker, start_date_str, end_date_str, progress_queue))
                p.start()
                processes.append(p)

            finished = set()
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
                        # run_strategy_for_ticker의 반환값을 받아 strategies에 추가
                        # (ProcessPoolExecutor가 아니라 multiprocessing.Process이므로 반환값 직접 수집 불가)
                        # 따라서, run_strategy_for_ticker에서 전략 객체를 Queue로 전달하도록 구조를 바꿔야 함
                except Exception:
                    pass
            for p in processes:
                p.join()
            for bar in bars.values():
                bar.close()

            # 각 티커별로 run_strategy_for_ticker의 결과를 받아 strategies에 추가
            # (Queue를 통해 전략 객체를 전달받는 구조가 아니라면, run_strategy_for_ticker에서 전략 객체를 파일로 저장 후 불러오거나,
            # 또는 run_strategy_for_ticker 내에서 전략 성과를 별도 Queue로 전달하도록 구조를 개선해야 함)
            # 임시로, run_strategy_for_ticker에서 전략 객체를 pickle로 저장하고, 여기서 불러오는 방식 예시:
            import glob, pickle
            result_files = glob.glob('strategy_result_*.pkl')
            for rf in result_files:
                try:
                    with open(rf, 'rb') as f:
                        strategy = pickle.load(f)
                        if strategy is not None:
                            strategies.append(strategy)
                except Exception:
                    pass
                try:
                    os.remove(rf)
                except Exception:
                    pass

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
            all_results.extend([r.to_dict() for r in backtest_results])
            # 결과 저장 및 분석
            save_backtest_results(backtest_results)
            analyze_backtest_history()
            logger.info(f"\n{LogEmoji.SUCCESS} 백테스트 완료")
        # 반복 결과 종합 분석
        import pandas as pd
        if all_results:
            df_all = pd.DataFrame(all_results)
            import os
            summary_path = os.path.abspath('results_summary.txt')
            try:
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write('[반복 백테스트] 코인별 평균/표준편차 성과\n')
                    coin_summary = df_all.groupby('ticker').agg({
                        'total_trades': ['mean', 'std'],
                        'win_rate': ['mean', 'std'],
                        'total_profit': ['mean', 'std'],
                        'avg_profit': ['mean', 'std']
                    }).round(2)
                    f.write(str(coin_summary))
                    f.write('\n\n[반복 백테스트] 시장상태별 평균/표준편차 성과\n')
                    market_summary = df_all.groupby('market_condition').agg({
                        'total_trades': ['mean', 'std'],
                        'win_rate': ['mean', 'std'],
                        'total_profit': ['mean', 'std'],
                        'avg_profit': ['mean', 'std']
                    }).round(2)
                    f.write(str(market_summary))
                print(f'요약 결과가 {summary_path} 파일에 저장되었습니다.')
                print('\n[반복 백테스트] 코인별 평균/표준편차 성과')
                print(coin_summary)
                print('\n[반복 백테스트] 시장상태별 평균/표준편차 성과')
                print(market_summary)
            except Exception as e:
                print(f'요약 파일 저장 중 오류: {e}')
        # ====== market_debug.log 자동 정제 및 분포 출력 ======
        import os
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'market_debug.log')
        clean_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'market_debug_clean.log')
        with open(log_path, 'r', encoding='utf-8', errors='replace') as fin, open(clean_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                if line.count(',') == 6:
                    fout.write(line)
        try:
            import pandas as pd
            log = pd.read_csv(clean_path, header=None,
                              names=['datetime', 'ticker', 'trade_type', 'price', 'profit_pct', 'reason', 'market_condition'],
                              encoding='utf-8')
            log['date'] = pd.to_datetime(log['datetime'], errors='coerce').dt.date
            exits = log[log['trade_type'] == 'EXIT'].copy()
            report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report_summary.txt')
            with open(report_path, 'w', encoding='utf-8') as rf:
                rf.write('[시장상태별 EXIT 시점 성과]\n')
                market_report = exits.groupby('market_condition')['profit_pct'].agg(['mean', 'std', 'count'])
                rf.write(str(market_report)+'\n\n')
                print('\n[시장상태별 EXIT 시점 성과]')
                print(market_report)
                rf.write('[시장상태별 EXIT 시점 승률]\n')
                win_report = exits.groupby('market_condition').apply(lambda x: (x['profit_pct'] > 0).mean())
                rf.write(str(win_report)+'\n\n')
                print('\n[시장상태별 EXIT 시점 승률]')
                print(win_report)
                rf.write('[코인별 EXIT 시점 성과]\n')
                coin_report = exits.groupby('ticker')['profit_pct'].agg(['mean', 'std', 'count'])
                rf.write(str(coin_report)+'\n\n')
                print('\n[코인별 EXIT 시점 성과]')
                print(coin_report)
                rf.write('[진입/청산 사유별 성과]\n')
                reason_report = exits.groupby('reason')['profit_pct'].agg(['mean', 'std', 'count'])
                rf.write(str(reason_report)+'\n\n')
                print('\n[진입/청산 사유별 성과]')
                print(reason_report)
                rf.write('[날짜별 EXIT 시점 성과]\n')
                date_report = exits.groupby('date')['profit_pct'].agg(['mean', 'std', 'count'])
                rf.write(str(date_report.tail(10))+'\n\n')
                print('\n[최근 10일간 EXIT 시점 성과]')
                print(date_report.tail(10))
            print(f'리포트가 {report_path} 파일에 저장되었습니다.')
        except Exception as e:
            print(f'시장상태별 성과 리포트 오류: {e}')
    except Exception as e:
        logger.error(f"{LogEmoji.ERROR} 프로그램 실행 중 오류: {str(e)}")
