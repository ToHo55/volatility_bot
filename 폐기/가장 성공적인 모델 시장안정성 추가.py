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
import concurrent.futures
from collections import defaultdict

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
    logger.setLevel(logging.WARNING)  # WARNING 레벨로 변경하여 오류 메시지 숨김
    
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

class ProgressBar:
    """멀티프로세스 환경에서 사용할 프로그레스 바"""
    def __init__(self, total: int, prefix: str = '', length: int = 30):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.current = 0
        self.start_time = time.time()
        
    def update(self, current: int):
        """프로그레스 바 업데이트"""
        self.current = current
        percent = (current / self.total) * 100
        filled_length = int(self.length * current // self.total)
        bar = '█' * filled_length + '-' * (self.length - filled_length)
        
        # 예상 남은 시간 계산
        elapsed_time = time.time() - self.start_time
        if current > 0:
            estimated_total = elapsed_time / (current / self.total)
            remaining_time = max(0, estimated_total - elapsed_time)
            minutes = int(remaining_time // 60)
            seconds = int(remaining_time % 60)
            time_str = f"{minutes}분 " if minutes > 0 else ""
            time_str += f"{seconds}초"
        else:
            time_str = "계산 중..."
        
        # 프로그레스 바 출력 (캐리지 리턴 없이)
        print(f'\033[K{self.prefix} |{bar}| {percent:.1f}% 남은시간: {time_str}')

def print_progress(progress_data):
    """여러 코인의 진행 상황을 동시에 표시"""
    # 터미널 화면 지우기
    print('\033[2J\033[H', end='')
    
    # 기본 정보 출력
    print(f"백테스트 진행 상황:")
    print("-" * 60)
    
    # 각 코인별 진행 상황 출력
    for ticker, data in progress_data.items():
        trades = data['trades']
        profit = data['profit']
        percent = data['percent']
        bar_length = 30
        filled_length = int(bar_length * percent / 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"{ticker:<10} |{bar}| {percent:5.1f}% | 거래: {trades:3d}건 | 수익: {profit:+6.2f}%")
    
    print("-" * 60)

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
    """데이터 수집 진행률을 출력합니다."""
    percent = (current / total) * 100
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent:5.1f}% - {month} 수집 중', end='', flush=True)

def fetch_ohlcv(ticker: str, start_date: str, end_date: str, progress_dict=None, data_loading_progress=None) -> Optional[pd.DataFrame]:
    """OHLCV 데이터 조회 (페이지네이션 및 캐시 시스템 개선)"""
    try:
        if data_loading_progress is not None:
            data_loading_progress[ticker] = {'loaded': False, 'progress': 0}
        
        logger.info(f"📊 {ticker} 데이터 로드 중...")
        
        # 날짜 변환
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 캐시 파일 경로 설정 (월별 캐시)
        cache_files = []
        current_month = start_dt.replace(day=1)
        total_months = 0
        while current_month <= end_dt:
            cache_key = f"{ticker}_{current_month.strftime('%Y%m')}"
            cache_file = CACHE_DIR / f"{cache_key}.pkl"
            cache_files.append((current_month, cache_file))
            total_months += 1
            current_month = (current_month + timedelta(days=32)).replace(day=1)
        
        # 캐시된 데이터 확인 및 로드
        cached_data = []
        missing_months = []
        loaded_months = 0
        
        for month_start, cache_file in cache_files:
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                        if 'data' in cache_data and len(cache_data['data']) > 0:
                            cached_data.append(cache_data['data'])
                            loaded_months += 1
                            if data_loading_progress is not None:
                                progress = (loaded_months / total_months) * 100
                                data_loading_progress[ticker] = {'loaded': False, 'progress': progress}
                            continue
                except Exception as e:
                    logger.warning(f"캐시 파일 로드 실패 ({cache_file.name})")
            
            missing_months.append((month_start, cache_file))
        
        # 누락된 데이터 수집
        for month_start, cache_file in missing_months:
            month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            month_str = month_start.strftime('%Y-%m')
            
            # 해당 월의 데이터 수집
            month_data = []
            current_to = min(month_end, end_dt)
            total_days = (month_end - month_start).days + 1
            processed_days = 0
            
            while current_to >= month_start:
                try:
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
                            
                            # 진행률 업데이트
                            processed_days = (month_end - current_to).days
                            month_progress = (processed_days / total_days) * 100
                            total_progress = ((loaded_months + month_progress/100) / total_months) * 100
                            
                            if data_loading_progress is not None:
                                data_loading_progress[ticker] = {
                                    'loaded': False,
                                    'progress': total_progress
                                }
                            
                            time.sleep(0.1)
                            break
                            
                        except Exception as e:
                            if retry < 4:
                                time.sleep(1.0 * (2 ** retry))
                            else:
                                break
                except KeyboardInterrupt:
                    logger.warning(f"\n{ticker} 데이터 수집이 중단되었습니다.")
                    return None
            
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
                loaded_months += 1
                
                if data_loading_progress is not None:
                    progress = (loaded_months / total_months) * 100
                    data_loading_progress[ticker] = {'loaded': False, 'progress': progress}
        
        if not cached_data:
            logger.error(f"❌ {ticker}: 데이터 수집 실패")
            return None
        
        # 전체 데이터 결합 및 정리
        df = pd.concat(cached_data)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        
        # 요청 기간에 맞게 필터링
        df = df[start_date:end_date]
        
        if len(df) > 0:
            if data_loading_progress is not None:
                data_loading_progress[ticker] = {'loaded': True, 'progress': 100}
            logger.info(f"✅ {ticker} 데이터 로드 완료 ({len(df):,}개)")
            return df
        else:
            logger.error(f"❌ {ticker}: 데이터가 비어있음")
            return None
        
    except Exception as e:
        logger.error(f"❌ {ticker}: 데이터 조회 실패")
        return None

class MarketCondition(Enum):
    STRONG_UPTREND = "강한 상승추세"
    WEAK_UPTREND = "약한 상승추세"
    SIDEWAYS = "횡보장"
    WEAK_DOWNTREND = "약한 하락추세"
    STRONG_DOWNTREND = "강한 하락추세"
    HIGH_VOLATILITY = "고변동성"
    LOW_VOLATILITY = "저변동성"
    ENTRY = "진입"
    EXIT = "청산"
    UNKNOWN = "알 수 없음"  # 추가된 상태

    @classmethod
    def get_safe_value(cls, value: str) -> 'MarketCondition':
        """안전하게 MarketCondition 값을 반환"""
        try:
            return cls(value)
        except ValueError:
            return cls.UNKNOWN

@dataclass
class TradeMetrics:
    trade_count: int = 0
    total_trades: int = 0
    win_rate: float = 0.0
    total_profit: float = 0.0
    avg_profit: float = 0.0
    market_condition: str = "횡보장"

    def to_dict(self) -> dict:
        return {
            'trade_count': self.trade_count,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'total_profit': self.total_profit,
            'avg_profit': self.avg_profit,
            'market_condition': self.market_condition
        }

@dataclass
class TradeRecord:
    timestamp: datetime
    type: str  # 'ENTRY' or 'EXIT'
    price: float
    profit_pct: float = 0.0
    success: bool = False
    market_condition: str = MarketCondition.UNKNOWN.value

    def __post_init__(self):
        try:
            # 타임스탬프 처리
            if isinstance(self.timestamp, str):
                self.timestamp = pd.to_datetime(self.timestamp, errors='coerce')
                if pd.isna(self.timestamp):
                    self.timestamp = datetime.now()
            elif not isinstance(self.timestamp, datetime):
                self.timestamp = datetime.now()

            # 가격 처리
            if isinstance(self.price, str):
                # 쉼표 제거 및 숫자 변환
                price_str = self.price.replace(',', '').strip()
                try:
                    self.price = float(price_str)
                except ValueError:
                    logger.error(f"가격 변환 오류: {self.price}")
                    self.price = 0.0
            else:
                try:
                    self.price = float(self.price)
                except (ValueError, TypeError):
                    logger.error(f"가격 변환 오류: {self.price}")
                    self.price = 0.0

            # 수익률 처리
            if isinstance(self.profit_pct, str):
                try:
                    # 퍼센트 기호 및 쉼표 제거
                    profit_str = self.profit_pct.replace('%', '').replace(',', '').strip()
                    self.profit_pct = float(profit_str)
                except ValueError:
                    logger.error(f"수익률 변환 오류: {self.profit_pct}")
                    self.profit_pct = 0.0
            else:
                try:
                    self.profit_pct = float(self.profit_pct)
                except (ValueError, TypeError):
                    logger.error(f"수익률 변환 오류: {self.profit_pct}")
                    self.profit_pct = 0.0

            # 거래 타입 검증
            if self.type not in ['ENTRY', 'EXIT']:
                logger.warning(f"잘못된 거래 타입: {self.type}")
                self.type = 'ENTRY'

            # 시장 상태 검증
            if not self.market_condition or self.market_condition not in [mc.value for mc in MarketCondition]:
                self.market_condition = MarketCondition.UNKNOWN.value

        except Exception as e:
            logger.error(f"TradeRecord 초기화 중 오류: {str(e)}")
            # 기본값 설정
            self.timestamp = datetime.now()
            self.price = 0.0
            self.profit_pct = 0.0
            self.type = 'ENTRY'
            self.success = False
            self.market_condition = MarketCondition.UNKNOWN.value

    def to_dict(self) -> dict:
        """거래 기록을 딕셔너리로 변환"""
        try:
            return {
                'timestamp': self.timestamp,
                'type': self.type,
                'price': self.price,
                'profit_pct': self.profit_pct,
                'success': self.success,
                'market_condition': self.market_condition
            }
        except Exception as e:
            logger.error(f"거래 기록 변환 중 오류: {str(e)}")
            return {
                'timestamp': datetime.now(),
                'type': 'UNKNOWN',
                'price': 0.0,
                'profit_pct': 0.0,
                'success': False,
                'market_condition': MarketCondition.UNKNOWN.value
            }

class TradingSummary:
    def __init__(self):
        self.trades: List[TradeRecord] = []
        self.total_trades = 0
        self.successful_trades = 0
        self.total_profit = 0.0
        self.win_rate = 0.0
        self.market_conditions = defaultdict(int)
        self.last_update = datetime.now()

    def add_trade(self, trade: TradeRecord) -> None:
        """거래 기록 추가 및 통계 업데이트"""
        try:
            if not isinstance(trade, TradeRecord):
                logger.error(f"잘못된 거래 기록 형식: {type(trade)}")
                return

            self.trades.append(trade)
            self.total_trades += 1
            
            if trade.success:
                self.successful_trades += 1
            
            try:
                self.total_profit += float(trade.profit_pct)
            except (ValueError, TypeError) as e:
                logger.error(f"수익률 계산 오류: {str(e)}")
            
            if self.total_trades > 0:
                self.win_rate = (self.successful_trades / self.total_trades) * 100
            
            self.market_conditions[trade.market_condition] += 1
            self.last_update = datetime.now()

        except Exception as e:
            logger.error(f"거래 기록 추가 중 오류: {str(e)}")

    def get_summary(self) -> dict:
        """거래 요약 정보 반환"""
        try:
            return {
                'total_trades': self.total_trades,
                'successful_trades': self.successful_trades,
                'total_profit': round(self.total_profit, 2),
                'win_rate': round(self.win_rate, 2),
                'market_conditions': dict(self.market_conditions),
                'last_update': self.last_update.strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logger.error(f"거래 요약 생성 중 오류: {str(e)}")
            return {
                'total_trades': 0,
                'successful_trades': 0,
                'total_profit': 0.0,
                'win_rate': 0.0,
                'market_conditions': {},
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    def reset(self) -> None:
        """모든 통계 초기화"""
        try:
            self.trades.clear()
            self.total_trades = 0
            self.successful_trades = 0
            self.total_profit = 0.0
            self.win_rate = 0.0
            self.market_conditions.clear()
            self.last_update = datetime.now()
        except Exception as e:
            logger.error(f"통계 초기화 중 오류: {str(e)}")

    def get_trade_history(self) -> List[dict]:
        """전체 거래 기록 반환"""
        try:
            return [trade.to_dict() for trade in self.trades]
        except Exception as e:
            logger.error(f"거래 기록 변환 중 오류: {str(e)}")
            return []

    def calculate_metrics(self) -> TradeMetrics:
        """상세 거래 지표 계산 (TradeMetrics 객체로 반환)"""
        try:
            if not self.trades:
                return TradeMetrics()

            profits = [t.profit_pct for t in self.trades]
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [p for p in profits if p < 0]

            trade_count = len(self.trades)
            win_rate = (len(winning_trades) / trade_count) * 100 if trade_count > 0 else 0.0
            total_profit = sum(profits)
            avg_profit = sum(winning_trades) / len(winning_trades) if winning_trades else 0.0
            market_condition = self.trades[-1].market_condition if self.trades else "횡보장"

            metrics = TradeMetrics(
                trade_count=trade_count,
                total_trades=trade_count,
                win_rate=win_rate,
                total_profit=total_profit,
                avg_profit=avg_profit,
                market_condition=market_condition
            )
            return metrics
        except Exception as e:
            logger.error(f"지표 계산 중 오류: {str(e)}")
            return TradeMetrics()

    def _get_max_consecutive(self, is_win: bool) -> int:
        """최대 연속 승/패 계산"""
        try:
            current = 0
            max_consecutive = 0
            
            for trade in self.trades:
                if trade.success == is_win:
                    current += 1
                    max_consecutive = max(max_consecutive, current)
                else:
                    current = 0
                    
            return max_consecutive
            
        except Exception as e:
            logger.error(f"연속 승/패 계산 중 오류: {str(e)}")
            return 0

class StrategyConfig:
    """전략 설정 클래스"""
    def __init__(self, is_random: bool = False):
        # 기본 파라미터 (고정값)
        self.base_rsi_period = 14
        self.base_bb_period = 20
        self.base_bb_std = 2.0
        self.base_volume_period = 20
        self.base_trend_period = 20
        
        # RSI 진입 임계값 (기본값)
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        # 가격 변화율 기준
        self.price_change_threshold = -0.003
        
        # 거래량 급등 기준
        self.volume_surge_threshold = 2.0
        
        # 변동성 기준
        self.volatility_threshold = 0.015
        
        # 랜덤화 여부
        self.is_random = is_random

    def _get_fixed_params(self, ticker: str) -> dict:
        """코인별 최적화된 파라미터 반환"""
        # 메이저 코인 (BTC, ETH) - 더 보수적이고 엄격한 조건
        if 'BTC' in ticker:
            return {
                'rsi_period': 14,          # RSI 기간 증가
                'bb_period': 20,           # 볼린저 밴드 기간 증가
                'bb_std': 2.2,             # 표준편차 증가로 더 엄격한 진입
                'volatility_threshold': 0.012,  # 변동성 기준 감소
                'volume_threshold': 1.5,    # 거래량 기준 증가
                'rsi_oversold': 25,        # 과매도 기준 강화
                'rsi_overbought': 75,      # 과매수 기준 강화
                'profit_target': 0.015,    # 목표 수익 1.5%
                'stop_loss': 0.01,         # 손절 기준 1%
                'min_volume_ratio': 1.2    # 최소 거래량 비율
            }
        elif 'ETH' in ticker:
            return {
                'rsi_period': 14,
                'bb_period': 20,
                'bb_std': 2.1,
                'volatility_threshold': 0.013,
                'volume_threshold': 1.4,
                'rsi_oversold': 27,
                'rsi_overbought': 73,
                'profit_target': 0.016,
                'stop_loss': 0.011,
                'min_volume_ratio': 1.15
            }
        # 알트코인 - 메이저 코인 기준으로 조정된 파라미터
        else:
            return {
                'rsi_period': 12,          # 빠른 대응을 위해 기간 감소
                'bb_period': 18,           # 볼린저 밴드 기간 소폭 감소
                'bb_std': 2.3,             # 더 엄격한 진입을 위해 표준편차 증가
                'volatility_threshold': 0.015,  # 변동성 기준 소폭 상향
                'volume_threshold': 1.6,    # 거래량 기준 강화
                'rsi_oversold': 28,        # 과매도 기준 강화
                'rsi_overbought': 72,      # 과매수 기준 강화
                'profit_target': 0.018,    # 목표 수익 1.8%
                'stop_loss': 0.012,        # 손절 기준 1.2%
                'min_volume_ratio': 1.3    # 최소 거래량 비율 증가
            }

    def get_params(self, ticker: str) -> dict:
        """코인별 최적화된 파라미터 반환"""
        if self.is_random:
            return self._get_random_params(ticker)
        else:
            return self._get_fixed_params(ticker)

    def _get_random_params(self, ticker: str) -> dict:
        """랜덤화된 파라미터 반환 (기본 파라미터 기준 ±20% 변동)"""
        base_params = self._get_fixed_params(ticker)
        return {
            'rsi_period': random.randint(
                int(base_params['rsi_period'] * 0.8),
                int(base_params['rsi_period'] * 1.2)
            ),
            'bb_period': random.randint(
                int(base_params['bb_period'] * 0.8),
                int(base_params['bb_period'] * 1.2)
            ),
            'bb_std': random.uniform(
                base_params['bb_std'] * 0.8,
                base_params['bb_std'] * 1.2
            ),
            'volatility_threshold': base_params['volatility_threshold'],
            'volume_threshold': base_params['volume_threshold'],
            'rsi_oversold': base_params['rsi_oversold'],
            'rsi_overbought': base_params['rsi_overbought'],
            'profit_target': base_params['profit_target'],
            'stop_loss': base_params['stop_loss'],
            'min_volume_ratio': base_params['min_volume_ratio']
        }

class MarketFilter:
    """시장 상태 필터링"""
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.trend_threshold = 0.06  # 8% -> 6%로 완화
        self.volume_surge = 1.1  # 1.2 -> 1.1로 완화
        self.volatility_threshold = 0.02  # 0.025 -> 0.02로 완화

    def check_market_conditions(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """시장 상태 체크 (변동성, 추세 등) - 결측치/컬럼 보정 및 UNKNOWN 최소화"""
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    return False, "컬럼 누락"
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # 결측치 보간
            df = df.fillna(method='ffill').dropna()
            if len(df) < 60:
                return False, "충분한 데이터가 없습니다"
            # 데이터 타입 변환 시도
            try:
                current_price = float(df['close'].iloc[-1])
            except Exception as e:
                return False, MarketCondition.UNKNOWN.value
            # 데이터프레임 복사 및 전처리
            df = df.copy()
            # 이동평균
            ma20 = df['close'].ewm(span=20, adjust=False).mean()
            ma60 = df['close'].ewm(span=60, adjust=False).mean()
            # 볼린저 밴드
            bb_period = 20
            rolling_mean = df['close'].rolling(window=bb_period).mean()
            rolling_std = df['close'].rolling(window=bb_period).std()
            upper_band = rolling_mean + (rolling_std * 2)
            lower_band = rolling_mean - (rolling_std * 2)
            # ATR 계산
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            # 최종값 추출
            current_ma20 = float(ma20.iloc[-1])
            current_ma60 = float(ma60.iloc[-1])
            current_upper = float(upper_band.iloc[-1])
            current_lower = float(lower_band.iloc[-1])
            current_atr = float(atr.iloc[-1])
            # 거래량 분석
            volume_sma = df['volume'].rolling(window=20).mean()
            current_volume = float(df['volume'].iloc[-1])
            current_volume_sma = float(volume_sma.iloc[-1])
            volume_ratio = current_volume / current_volume_sma if current_volume_sma > 0 else 0
            # 상태 판단
            conditions = []
            if current_price > current_upper and volume_ratio > 1.5:
                conditions.append("과매수")
            if current_price < current_lower and volume_ratio > 1.5:
                conditions.append("과매도")
            if current_atr > current_price * 0.02:
                conditions.append("고변동성")
            if volume_ratio < 0.5:
                conditions.append("저거래량")
            if conditions:
                return False, " & ".join(conditions)
            # 추세 판단
            if current_ma20 > current_ma60 * 1.02:
                return True, MarketCondition.WEAK_UPTREND.value
            elif current_ma20 < current_ma60 * 0.98:
                return True, MarketCondition.WEAK_DOWNTREND.value
            else:
                return True, MarketCondition.SIDEWAYS.value
        except Exception as e:
            return False, MarketCondition.UNKNOWN.value

class EntrySignalDetector:
    """진입 시그널 탐지"""
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.rsi_threshold_long = 35  # RSI 기준 완화 (45 -> 35)
        self.rsi_threshold_short = 65  # RSI 기준 완화 (55 -> 65)
        self.price_change_threshold = -0.0005  # 가격 변화율 완화 (-0.001 -> -0.0005)

    def detect_signal(self, df: pd.DataFrame) -> Tuple[bool, str, str]:
        """2차 필터: 진입 시그널 탐지"""
        try:
            # RSI 계산
            rsi = self.calculate_rsi(df)
            current_rsi = rsi.iloc[-1]
            
            # 가격 변화율 (5분으로 단축, 15분->5분)
            price_change = df['close'].pct_change(1).iloc[-1]
            
            # 진입 시그널 확인 (조건 완화)
            if current_rsi < self.rsi_threshold_long:
                return True, "LONG", "RSI 하향"
                
            if current_rsi > self.rsi_threshold_short:
                return True, "SHORT", "RSI 상향"
            
            return False, "", "시그널 없음"
            
        except Exception as e:
            logger.error(f"시그널 탐지 중 오류: {str(e)}")
            return False, "", "오류 발생"

    def calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """RSI(Relative Strength Index) 계산"""
        try:
            # 가격 변화 계산
            delta = df['close'].diff()
            
            # 상승/하락 구분
            gain = (delta.where(delta > 0, 0))
            loss = (-delta.where(delta < 0, 0))
            
            # 평균 계산 (0으로 나누기 방지)
            avg_gain = gain.rolling(window=self.rsi_period).mean()
            avg_loss = loss.rolling(window=self.rsi_period).mean()
            
            # RSI 계산 (0으로 나누기 방지)
            rs = avg_gain / avg_loss.replace(0, float('inf'))
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            # 오류 발생시 기본값 반환
            return pd.Series(50, index=df.index)  # 중립값 반환

class EntryConfirmer:
    """진입 안정성 확인"""
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.bb_period = 15  # 기간 단축 (20 -> 15)
        self.bb_std = 1.8    # 표준편차 완화 (2.0 -> 1.8)

    def confirm_entry(self, df: pd.DataFrame, entry_type: str) -> Tuple[bool, str]:
        """3차 필터: 진입 안정성 확인"""
        try:
            # 볼린저 밴드 계산
            bb_middle = df['close'].rolling(self.bb_period).mean()
            bb_std = df['close'].rolling(self.bb_period).std()
            bb_lower = bb_middle - (bb_std * self.bb_std)
            bb_upper = bb_middle + (bb_std * self.bb_std)
            
            # 현재값 추출 (스칼라 값으로 변환)
            current_price = safe_float(df['close'].iloc[-1])
            prev_close = safe_float(df['close'].iloc[-2])
            
            if entry_type == "LONG":
                # 볼린저 밴드 하단 근처 확인 (조건 완화)
                bb_lower_current = safe_float(bb_lower.iloc[-1])
                price_condition = current_price <= bb_lower_current * 1.02  # 2% 완화
                
                # 추가 확인봉 패턴 (단순화)
                pattern_condition = current_price > prev_close  # 양봉 전환만 확인
                
                if price_condition or pattern_condition:  # 조건 중 하나만 만족해도 진입
                    return True, "하단 돌파 또는 반등"
                    
            else:  # SHORT
                # 볼린저 밴드 상단 근처 확인 (조건 완화)
                bb_upper_current = safe_float(bb_upper.iloc[-1])
                price_condition = current_price >= bb_upper_current * 0.98  # 2% 완화
                
                # 추가 확인봉 패턴 (단순화)
                pattern_condition = current_price < prev_close  # 음봉 전환만 확인
                
                if price_condition or pattern_condition:  # 조건 중 하나만 만족해도 진입
                    return True, "상단 돌파 또는 하락"
            
            return False, "안정성 조건 미달"
            
        except Exception as e:
            logger.error(f"진입 확인 중 오류: {str(e)}")
            return False, "오류 발생"

class PositionSizer:
    """포지션 사이즈 계산"""
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.base_size = 0.15  # 기본 포지션 크기 증가 (0.1 -> 0.15)
        self.max_size = 0.3   # 최대 포지션 크기 유지
        self.min_size = 0.1  # 최소 포지션 크기 증가 (0.05 -> 0.1)

    def calculate_position_size(self, df: pd.DataFrame) -> float:
        """포지션 크기 동적 계산 (NaN 방어)"""
        try:
            volatility = safe_float(df['close'].pct_change().rolling(20).std().iloc[-1])
            vol_factor = 1.0 - (volatility / 0.05)
            volume_sma = df['volume'].rolling(20).mean()
            volume_ratio = safe_float(df['volume'].iloc[-1] / volume_sma.iloc[-1])
            vol_size_factor = min(volume_ratio, 2.0)
            position_size = self.base_size * vol_factor * vol_size_factor
            position_size = max(min(position_size, self.max_size), self.min_size)
            if pd.isna(position_size) or not np.isfinite(position_size):
                return self.min_size
            return float(position_size)
        except Exception as e:
            logger.error(f"포지션 크기 계산 중 오류: {str(e)}")
            return self.min_size

class EntryAnalyzer:
    """진입 조건 분석기"""
    def __init__(self):
        self.reset_stats()
    
    def reset_stats(self):
        self.total_checks = 0
        self.entry_success = 0
        self.condition_fails = {
            'rsi': 0,
            'bollinger': 0,
            'volume': 0,
            'price': 0
        }
        self.condition_stats = {
            'LONG': {
                'total': 0,
                'success': 0,
                'conditions': {
                    'rsi': 0,
                    'bollinger': 0,
                    'volume': 0,
                    'price': 0
                }
            },
            'SHORT': {
                'total': 0,
                'success': 0,
                'conditions': {
                    'rsi': 0,
                    'bollinger': 0,
                    'volume': 0,
                    'price': 0
                }
            }
        }
    
    def analyze_entry_conditions(self, conditions: List[bool], condition_names: List[str], position_type: str):
        """진입 조건 분석"""
        stats = self.condition_stats[position_type]
        stats['total'] += 1
        
        success = sum(conditions) >= 1  # 1개 이상 조건 충족 시 성공
        if success:
            stats['success'] += 1
            self.entry_success += 1
        
        for i, (condition, name) in enumerate(zip(conditions, condition_names)):
            if not condition:
                stats['conditions'][name] += 1
                self.condition_fails[name] += 1
        
        self.total_checks += 1
    
    def print_analysis(self, ticker: str):
        """분석 결과 출력"""
        logger.info(f"\n{LogEmoji.SUMMARY} {ticker} 진입 조건 분석:")
        logger.info(f"총 체크 횟수: {self.total_checks:,}회")
        logger.info(f"진입 성공률: {(self.entry_success/self.total_checks*100):.1f}%")
        
        logger.info("\n롱 포지션 분석:")
        long_stats = self.condition_stats['LONG']
        if long_stats['total'] > 0:
            logger.info(f"시도 횟수: {long_stats['total']:,}회")
            logger.info(f"성공률: {(long_stats['success']/long_stats['total']*100):.1f}%")
            logger.info("조건별 실패율:")
            for cond, fails in long_stats['conditions'].items():
                if long_stats['total'] > 0:
                    fail_rate = (fails/long_stats['total']*100)
                    logger.info(f"- {cond}: {fail_rate:.1f}%")
        
        logger.info("\n숏 포지션 분석:")
        short_stats = self.condition_stats['SHORT']
        if short_stats['total'] > 0:
            logger.info(f"시도 횟수: {short_stats['total']:,}회")
            logger.info(f"성공률: {(short_stats['success']/short_stats['total']*100):.1f}%")
            logger.info("조건별 실패율:")
            for cond, fails in short_stats['conditions'].items():
                if short_stats['total'] > 0:
                    fail_rate = (fails/short_stats['total']*100)
                    logger.info(f"- {cond}: {fail_rate:.1f}%")

class RSIReversalStrategy:
    def __init__(self, ticker: str, is_backtest: bool = False, randomize: bool = False):
        # 기존 초기화 코드 유지
        self.ticker = ticker
        self.is_backtest = is_backtest
        
        # 리스크 관리 파라미터 완화
        self.max_total_loss = -50.0  # 전체 손실 한도 (-30% -> -50%)
        self.max_consecutive_losses = 12  # 최대 연속 손실 횟수 (8 -> 12)
        self.current_consecutive_losses = 0
        self.total_profit_pct = 0.0
        
        # 변동성 필터 파라미터 완화
        self.volatility_lookback = 10  # 20 -> 10으로 단축
        self.max_volatility = 0.15  # 최대 허용 변동성 (8% -> 15%)
        
        # 추세 필터 파라미터 완화
        self.trend_lookback = 20  # 50 -> 20으로 단축
        self.trend_threshold = 0.005  # 추세 강도 임계값 (1% -> 0.5%)
        
        # 전략 설정 초기화
        self.config = StrategyConfig(is_random=randomize)
        
        # 필터 초기화
        self.market_filter = MarketFilter(self.config)
        self.signal_detector = EntrySignalDetector(self.config)
        self.entry_confirmer = EntryConfirmer(self.config)
        
        # 파라미터 설정
        params = self.config.get_params(ticker)
        self.rsi_period = params['rsi_period']
        self.bb_period = params['bb_period']
        self.bb_std = params['bb_std']
        self.volatility_threshold = params['volatility_threshold']
        self.volume_threshold = params['volume_threshold']
        
        # 진입/청산 임계값 완화
        self.rsi_entry_low = 40  # 35 -> 40
        self.rsi_entry_high = 60  # 65 -> 60
        self.price_change_threshold = -0.0003  # -0.0005 -> -0.0003
        
        # 포지션 관련 설정
        self.base_size = 0.15  # 15%
        self.max_size = 0.3    # 30%
        self.min_size = 0.1    # 10%
        
        # 현재 포지션 상태
        self.current_position = None
        self.entry_price = 0
        self.entry_time = None
        self.position_size = 0
        
        # 거래 기록
        self.trades = []
        self.summary = TradingSummary()
        
        # 성과 집계
        self.total_trades = 0
        self.successful_trades = 0
        self.total_profit_pct = 0.0
        
        # 진입 조건 분석기
        self.entry_analyzer = EntryAnalyzer()
        
        # 데이터프레임
        self.df = None
        self.last_trade_time = None  # 마지막 거래 시간 기록

    def check_risk_limits(self) -> Tuple[bool, str]:
        """리스크 한도 체크"""
        # 전체 손실 한도 체크
        if self.total_profit_pct <= self.max_total_loss:
            return False, f"전체 손실 한도 도달 ({self.total_profit_pct:.1f}%)"
            
        # 연속 손실 체크
        if self.current_consecutive_losses >= self.max_consecutive_losses:
            return False, f"연속 손실 한도 도달 ({self.current_consecutive_losses}회)"
            
        return True, ""

    _unknown_market_debug_count = 0  # 클래스 변수로 디버그 카운터 추가
    _unknown_market_debug_limit = 100
    def check_market_conditions(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """시장 상황 체크 (상세 로깅) - 반드시 Enum value만 반환, UNKNOWN 반환시 파일로 디버깅 로그 저장"""
        try:
            # 변동성 체크
            returns = df['close'].pct_change()
            volatility = safe_float(returns.rolling(self.volatility_lookback).std() * np.sqrt(252))
            # 추세 체크
            ma20 = df['close'].rolling(window=20).mean()
            ma50 = df['close'].rolling(window=50).mean()
            trend_strength = (ma20.iloc[-1] - ma50.iloc[-1]) / ma50.iloc[-1]
            # 거래량 체크
            volume_sma = df['volume'].rolling(10).mean()
            current_volume = safe_float(df['volume'].iloc[-1])
            volume_ratio = current_volume / safe_float(volume_sma.iloc[-1]) if safe_float(volume_sma.iloc[-1]) > 0 else 0
            # 데이터 부족 체크
            if len(df) < 60:
                if RSIReversalStrategy._unknown_market_debug_count < RSIReversalStrategy._unknown_market_debug_limit:
                    try:
                        with open('unknown_market_debug.log', 'a', encoding='utf-8') as f:
                            f.write(f'[디버그] 시장상태 UNKNOWN(데이터 부족): df len={len(df)}\n')
                    except Exception as file_e:
                        pass
                    RSIReversalStrategy._unknown_market_debug_count += 1
                return True, MarketCondition.UNKNOWN.value
            # 실제 시장상태 Enum value 반환
            if trend_strength > self.trend_threshold:
                return True, MarketCondition.WEAK_UPTREND.value
            elif trend_strength < -self.trend_threshold:
                return True, MarketCondition.WEAK_DOWNTREND.value
            else:
                return True, MarketCondition.SIDEWAYS.value
        except Exception as e:
            if RSIReversalStrategy._unknown_market_debug_count < RSIReversalStrategy._unknown_market_debug_limit:
                try:
                    with open('unknown_market_debug.log', 'a', encoding='utf-8') as f:
                        f.write(f'[디버그] 시장상태 UNKNOWN 반환: {e}, df len={len(df)}\n')
                except Exception as file_e:
                    pass
                RSIReversalStrategy._unknown_market_debug_count += 1
            return True, MarketCondition.UNKNOWN.value

    def evaluate_entry(self, df: pd.DataFrame) -> Tuple[bool, str, str]:
        """진입 조건 평가 (상세 로깅)"""
        try:
            # RSI 계산
            rsi = self.calculate_rsi(df)
            current_rsi = safe_float(rsi.iloc[-1])
            
            # 볼린저 밴드 계산
            bb_lower, bb_middle, bb_upper = self.calculate_bollinger_bands(df)
            current_price = safe_float(df['close'].iloc[-1])
            
            # 이동평균 계산
            ma5 = df['close'].rolling(window=5).mean()
            ma10 = df['close'].rolling(window=10).mean()
            
            logger.debug(f"{self.ticker} 진입 조건 평가:")
            logger.debug(f"- RSI: {current_rsi:.1f}")
            logger.debug(f"- 현재가: {current_price:,.0f}")
            logger.debug(f"- BB 중앙선: {safe_float(bb_middle.iloc[-1]):,.0f}")
            logger.debug(f"- MA5: {safe_float(ma5.iloc[-1]):,.0f}")
            logger.debug(f"- MA10: {safe_float(ma10.iloc[-1]):,.0f}")
            
            # LONG 진입 조건
            long_conditions = []
            if current_rsi < 45:  # RSI 기준 완화
                long_conditions.append(f"RSI({current_rsi:.1f})")
            if current_price < safe_float(bb_middle.iloc[-1]):
                long_conditions.append("BB중앙선")
            if float(ma5.iloc[-1]) < safe_float(ma10.iloc[-1]):
                long_conditions.append("MA하락")
            
            # SHORT 진입 조건
            short_conditions = []
            if current_rsi > 55:
                short_conditions.append(f"RSI({current_rsi:.1f})")
            if current_price > safe_float(bb_middle.iloc[-1]):
                short_conditions.append("BB중앙선")
            if float(ma5.iloc[-1]) > safe_float(ma10.iloc[-1]):
                short_conditions.append("MA상승")
            
            logger.debug(f"- LONG 조건: {', '.join(long_conditions) if long_conditions else '없음'}")
            logger.debug(f"- SHORT 조건: {', '.join(short_conditions) if short_conditions else '없음'}")
            
            # 진입 판단 (1개 이상 조건 만족)
            if len(long_conditions) >= 1:
                return True, "LONG", " | ".join(long_conditions)
            if len(short_conditions) >= 1:
                return True, "SHORT", " | ".join(short_conditions)
            
            return False, "", "조건 불충분"
            
        except Exception as e:
            logger.error(f"{self.ticker} 진입 평가 중 오류: {str(e)}")
            return False, "", f"오류: {str(e)}"

    def process_single_window(self, window: pd.DataFrame) -> List[Dict]:
        trades = []
        try:
            if len(window) < self.bb_period:
                return trades
            current_price = safe_float(window['close'].iloc[-1])
            current_time = window.index[-1]
            min_trade_interval = 12  # 1시간(12캔들)로 변경
            if self.last_trade_time is not None:
                minutes_since_last = (current_time - self.last_trade_time).total_seconds() / 300
                if minutes_since_last < min_trade_interval:
                    return trades
            if self.current_position is None:
                market_ok, market_status = self.check_market_conditions(window)
                if not market_ok:
                    return trades
                should_enter, position_type, entry_reason = self.evaluate_entry(window)
                # 진입 조건을 2개 이상 만족해야 진입
                if should_enter and entry_reason.count('|') + 1 >= 2:
                    self.position_size = self.calculate_position_size(window)
                    self.current_position = position_type
                    self.entry_price = current_price
                    self.entry_time = current_time
                    self.last_trade_time = current_time
                    trade_record = TradeRecord(
                        timestamp=current_time,
                        type="ENTRY",
                        price=current_price,
                        market_condition=market_status
                    )
                    trades.append(trade_record)
                    self.summary.add_trade(trade_record)
                    log_trade_to_file(f"{self.ticker} 진입: {position_type} @ {current_price:,.0f} ({entry_reason}) | 시장상태: {market_status} | 포지션크기: {self.position_size:.2%}")
            else:
                hold_time = (current_time - self.entry_time).total_seconds() / 300
                should_exit, exit_reason = self.evaluate_exit(window, self.current_position, self.entry_price, hold_time)
                if should_exit:
                    profit_pct = ((current_price - self.entry_price) / self.entry_price * 100)
                    if self.current_position == "SHORT":
                        profit_pct = -profit_pct
                    trade_record = TradeRecord(
                        timestamp=current_time,
                        type="EXIT",
                        price=current_price,
                        profit_pct=profit_pct,
                        success=(profit_pct > 0),
                        market_condition=market_status if 'market_status' in locals() else "알 수 없음"
                    )
                    trades.append(trade_record)
                    self.summary.add_trade(trade_record)
                    self.last_trade_time = current_time
                    if profit_pct < 0:
                        self.current_consecutive_losses += 1
                        log_trade_to_file(f"{self.ticker} 연속 손실: {self.current_consecutive_losses}회")
                    else:
                        self.current_consecutive_losses = 0
                    self.total_profit_pct += profit_pct
                    log_trade_to_file(f"{self.ticker}: 청산 - {self.current_position} @ {current_price:.2f} (수익률: {profit_pct:+.2f}%, {exit_reason})")
                    self.current_position = None
                    self.entry_price = 0
                    self.entry_time = None
                    self.position_size = 0
            return trades
        except Exception as e:
            logger.error(f"{self.ticker} 거래 처리 중 오류: {str(e)}")
            return trades

    def calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """RSI(Relative Strength Index) 계산"""
        try:
            # 가격 변화 계산
            delta = df['close'].diff()
            
            # 상승/하락 구분
            gain = (delta.where(delta > 0, 0))
            loss = (-delta.where(delta < 0, 0))
            
            # 평균 계산 (0으로 나누기 방지)
            avg_gain = gain.rolling(window=self.rsi_period).mean()
            avg_loss = loss.rolling(window=self.rsi_period).mean()
            
            # RSI 계산 (0으로 나누기 방지)
            rs = avg_gain / avg_loss.replace(0, safe_float('inf'))
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            # 오류 발생시 기본값 반환
            return pd.Series(50, index=df.index)  # 중립값 반환

    def calculate_bollinger_bands(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """볼린저 밴드 계산"""
        try:
            # 중심선 (20일 이동평균)
            middle = df['close'].rolling(window=self.bb_period).mean()
            
            # 표준편차 (0 방지)
            std = df['close'].rolling(window=self.bb_period).std()
            std = std.replace(0, df['close'].std())  # 0인 경우 전체 기간의 표준편차 사용
            
            # 상단/하단 밴드
            upper = middle + (std * self.bb_std)
            lower = middle - (std * self.bb_std)
            
            return lower, middle, upper
            
        except Exception as e:
            # 오류 발생시 현재가 기준으로 기본값 반환
            current_price = df['close'].iloc[-1]
            default_std = df['close'].std() or current_price * 0.02  # 표준편차가 0이면 현재가의 2% 사용
            return (
                pd.Series(current_price * 0.98, index=df.index),  # lower
                pd.Series(current_price, index=df.index),         # middle
                pd.Series(current_price * 1.02, index=df.index)   # upper
            )

    def calculate_dynamic_exits(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """동적 청산가격 계산 (개선된 버전)"""
        try:
            current_price = safe_float(df['close'].iloc[-1])
            
            # ATR 계산 (변동성 기반 스탑로스)
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = safe_float(true_range.rolling(window=14).mean().iloc[-1])
            
            # RSI 기반 동적 조정
            rsi = self.calculate_rsi(df)
            current_rsi = safe_float(rsi.iloc[-1])
            
            # MACD 기반 추세 강도
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - signal
            trend_strength = abs(safe_float(macd_hist.iloc[-1]))
            
            # 손절가 계산 (ATR 기반)
            atr_multiplier = 2.0
            if trend_strength > 0.002:  # 강한 추세
                atr_multiplier = 2.5
            elif current_rsi < 30 or current_rsi > 70:  # 과매수/과매도
                atr_multiplier = 1.5
                
            stop_loss = current_price - (atr * atr_multiplier) if self.current_position == "LONG" else current_price + (atr * atr_multiplier)
            
            # 익절가 계산 (리스크:리워드 = 1:2)
            take_profit_distance = abs(current_price - stop_loss) * 2
            take_profit = current_price + take_profit_distance if self.current_position == "LONG" else current_price - take_profit_distance
            
            # 트레일링 스탑 계산
            trailing_distance = atr * 3.0  # ATR의 3배
            trailing_stop = current_price - trailing_distance if self.current_position == "LONG" else current_price + trailing_distance
            
            # 손절폭 제한 (최대 2%)
            max_stop_loss_pct = 0.02
            if abs(stop_loss - current_price) / current_price > max_stop_loss_pct:
                stop_loss = current_price * (1 - max_stop_loss_pct) if self.current_position == "LONG" else current_price * (1 + max_stop_loss_pct)
            
            return safe_float(stop_loss), safe_float(take_profit), safe_float(trailing_stop)
            
        except Exception as e:
            logger.error(f"진입 평가 중 오류: {str(e)}")
            return False, ""

    def calculate_position_size(self, df: pd.DataFrame) -> float:
        """포지션 크기 동적 계산 (NaN 방어)"""
        try:
            volatility = float(df['close'].pct_change().rolling(20).std().iloc[-1])
            vol_factor = 1.0 - (volatility / 0.05)
            volume_sma = df['volume'].rolling(20).mean()
            volume_ratio = float(df['volume'].iloc[-1] / volume_sma.iloc[-1])
            vol_size_factor = min(volume_ratio, 2.0)
            position_size = self.base_size * vol_factor * vol_size_factor
            position_size = max(min(position_size, self.max_size), self.min_size)
            if pd.isna(position_size) or not np.isfinite(position_size):
                return self.min_size
            return float(position_size)
        except Exception as e:
            logger.error(f"포지션 크기 계산 중 오류: {str(e)}")
            return self.min_size

    def evaluate_exit(self, df: pd.DataFrame, position_type: str, entry_price: float, entry_time: datetime) -> Tuple[bool, str]:
        """청산 조건 평가 (상세 로깅)"""
        try:
            current_price = safe_float(df['close'].iloc[-1])
            current_time = pd.to_datetime(df.index[-1])
            
            # 보유 시간 계산 (datetime 객체로 변환하여 계산)
            if isinstance(entry_time, (str, float, int)):
                entry_time = pd.to_datetime(entry_time)
            holding_minutes = (current_time - entry_time).total_seconds() / 60
            
            # 손익률 계산
            profit_rate = ((current_price - entry_price) / entry_price * 100) * (1 if position_type == "LONG" else -1)
            
            # RSI 계산
            rsi = self.calculate_rsi(df)
            current_rsi = safe_float(rsi.iloc[-1])
            
            logger.debug(f"{self.ticker} 청산 조건 평가:")
            logger.debug(f"- 포지션: {position_type}")
            logger.debug(f"- 진입가: {entry_price:,.0f}")
            logger.debug(f"- 현재가: {current_price:,.0f}")
            logger.debug(f"- 손익률: {profit_rate:.1f}%")
            logger.debug(f"- 보유시간: {holding_minutes:.0f}분")
            logger.debug(f"- RSI: {current_rsi:.1f}")
            
            exit_conditions = []
            
            # 손절 조건 (-2%)
            if profit_rate < -2:
                exit_conditions.append(f"손절({profit_rate:.1f}%)")
            
            # 익절 조건 (+1%)
            if profit_rate > 1:
                exit_conditions.append(f"익절({profit_rate:.1f}%)")
            
            # 최대 보유 시간 (60분)
            if holding_minutes > 60:
                exit_conditions.append(f"시간초과({holding_minutes:.0f}분)")
            
            # RSI 기반 청산
            if position_type == "LONG" and current_rsi > 65:
                exit_conditions.append(f"RSI과매수({current_rsi:.1f})")
            elif position_type == "SHORT" and current_rsi < 35:
                exit_conditions.append(f"RSI과매도({current_rsi:.1f})")
            
            logger.debug(f"- 청산 조건: {', '.join(exit_conditions) if exit_conditions else '없음'}")
            
            if exit_conditions:
                return True, " | ".join(exit_conditions)
            
            return False, "조건 불충분"
            
        except Exception as e:
            logger.error(f"{self.ticker} 청산 평가 중 오류: {str(e)}")
            return False, f"오류: {str(e)}"

    def update_progress(self, current_idx: int, total_size: int):
        """진행 상황 업데이트"""
        now = datetime.now()
        if (now - self.last_progress_update) >= self.progress_update_interval:
            self.summary.current_progress = (current_idx / total_size) * 100
            self.summary.last_update = now
            
            # 진행률 계산
            percent = (current_idx / total_size) * 100
            trades = self.summary.total_trades
            profit = self.summary.total_profit_pct
            
            # progress_dict 업데이트
            progress_data = {
                'progress': percent,
                'trades': trades,
                'profit': profit,
                'remaining_time': ((total_size - current_idx) / (current_idx + 1)) * (now - self.start_time).total_seconds()
            }
            
            # 전역 진행 상황 딕셔너리 업데이트
            if not hasattr(self, '_progress_dict'):
                self._progress_dict = {}
            self._progress_dict[self.ticker] = progress_data
            
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
        """전략 성과 요약 출력 (간소화)"""
        try:
            if not self.trades:
                return
                
            # 성과 지표 계산
            total_trades = len([t for t in self.trades if t['type'] == 'EXIT'])
            if total_trades == 0:
                return
                
            successful_trades = len([t for t in self.trades if t['type'] == 'EXIT' and t.get('profit_pct', 0) > 0])
            win_rate = (successful_trades / total_trades) * 100
            
            profits = [t.get('profit_pct', 0) for t in self.trades if t['type'] == 'EXIT']
            total_profit = sum(profits)
            avg_profit = total_profit / total_trades if total_trades > 0 else 0
            
            # 결과 출력 (간소화)
            print(f"\n{self.ticker} 성과 요약:")
            print(f"거래: {total_trades}건 | 승률: {win_rate:.1f}% | 총수익: {total_profit:+.2f}% | 평균수익: {avg_profit:+.2f}%")
            
        except Exception as e:
            logger.error(f"성과 요약 출력 중 오류: {str(e)}")

    def _evaluate_performance_grade(self, metrics: TradeMetrics) -> str:
        """성과 등급 평가 (종합적인 평가 시스템)"""
        try:
            # 점수 초기화
            total_score = 0
            
            # 1. 승률 기준 (40% 가중치)
            if metrics.win_rate >= 65:  # 65% 이상
                total_score += 40
            elif metrics.win_rate >= 60:  # 60% 이상
                total_score += 32
            elif metrics.win_rate >= 55:  # 55% 이상
                total_score += 24
            elif metrics.win_rate >= 50:  # 50% 이상
                total_score += 16
            else:
                total_score += 8
            
            # 2. 총 수익률 기준 (40% 가중치)
            if metrics.total_profit >= 30:  # 30% 이상
                total_score += 40
            elif metrics.total_profit >= 20:  # 20% 이상
                total_score += 32
            elif metrics.total_profit >= 10:  # 10% 이상
                total_score += 24
            elif metrics.total_profit >= 0:  # 수익
                total_score += 16
            else:  # 손실
                total_score += 8
            
            # 3. 거래 횟수 기준 (20% 가중치)
            if metrics.trade_count >= 500:  # 충분한 거래량
                total_score += 20
            elif metrics.trade_count >= 300:
                total_score += 16
            elif metrics.trade_count >= 100:
                total_score += 12
            else:  # 거래량 부족
                total_score += 8
            
            # 최종 등급 결정
            if total_score >= 90:
                return "S"
            elif total_score >= 80:
                return "A"
            elif total_score >= 70:
                return "B"
            elif total_score >= 60:
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
            
            # 완료된 거래만 필터링
            completed_trades = [t for t in self.summary.trades if t.type == 'EXIT']
            if not completed_trades:
                return metrics
            
            # 기본 지표 계산
            metrics.trade_count = len(completed_trades)
            successful_trades = len([t for t in completed_trades if t.success])
            metrics.win_rate = (successful_trades / metrics.trade_count) * 100
            
            # 수익성 지표
            profits = [t.profit_pct for t in completed_trades]
            gains = [p for p in profits if p > 0]
            losses = [p for p in profits if p <= 0]
            
            metrics.total_profit = sum(profits)  # 총 수익률
            metrics.total_trades = len(completed_trades)  # 총 거래 횟수
            metrics.successful_trades = successful_trades  # 성공한 거래 횟수
            
            metrics.avg_profit = sum(gains) / len(gains) if gains else 0
            metrics.avg_profit = sum(gains) / len(gains) if gains else 0
            metrics.avg_loss = sum(losses) / len(losses) if losses else 0
            metrics.total_profit = sum(profits)
            
            # 프로핏 팩터
            total_gains = sum(gains) if gains else 0
            total_losses = abs(sum(losses)) if losses else 0
            metrics.profit_factor = total_gains / total_losses if total_losses != 0 else float('inf')
            
            # 최대 낙폭
            cumulative = np.cumsum(profits)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = running_max - cumulative
            metrics.max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0
            
            # 시장 상태
            metrics.market_condition = MarketCondition(max(set(t.market_condition for t in completed_trades), key=completed_trades.count))
            
            return metrics
            
        except Exception as e:
            return TradeMetrics()

    def should_trade(self, window: pd.DataFrame) -> bool:
        # 최소 거래 간격 설정
        min_trade_interval = {
            "KRW-BTC": 12,  # 1시간
            "KRW-ETH": 6,   # 30분
            "KRW-DOGE": 3   # 15분
        }.get(self.ticker, 6)
        
        # 마지막 거래로부터의 시간 확인
        if self.last_trade_time:
            time_since_last_trade = (window.index[-1] - self.last_trade_time).total_seconds() / 300  # 5분 단위
            if time_since_last_trade < min_trade_interval:
                return False
        
        return True

def compare_strategies(strategies: List[RSIReversalStrategy]) -> pd.DataFrame:
    """전략 비교 결과 출력"""
    try:
        results = []
        for strategy in strategies:
            if not isinstance(strategy, RSIReversalStrategy):
                logger.warning(f"잘못된 전략 객체 타입: {type(strategy)}")
                continue
            metrics = strategy.calculate_metrics()
            if not isinstance(metrics, TradeMetrics):
                logger.warning(f"잘못된 메트릭스 객체 타입: {type(metrics)}")
                # dict라면 TradeMetrics로 변환
                if isinstance(metrics, dict):
                    metrics = TradeMetrics(**metrics)
                else:
                    continue
            win_rate = round(metrics.win_rate, 2)
            total_profit = round(metrics.total_profit, 2)
            avg_profit = round(metrics.avg_profit, 2)
            # market_condition이 Enum 객체면 value로 변환
            market_condition = metrics.market_condition
            if hasattr(market_condition, 'value'):
                market_condition = market_condition.value
            results.append({
                '티커': strategy.ticker,
                '시장상태': market_condition,
                '승률': f"{win_rate:.1f}%",
                '총수익': f"{total_profit:.2f}%",
                '평균수익': f"{avg_profit:.2f}%",
                '거래횟수': metrics.trade_count,
                '등급': strategy._evaluate_performance_grade(metrics)
            })
        if not results:
            logger.warning("비교할 전략 결과가 없습니다.")
            return pd.DataFrame()
        df_results = pd.DataFrame(results)
        print("\n" + "="*100)
        print("📊 전략 성과 비교")
        print("-"*100)
        print(df_results.to_string(index=False))
        print("="*100 + "\n")
        return df_results
    except Exception as e:
        logger.error(f"전략 비교 중 오류: {str(e)}")
        return pd.DataFrame()

def analyze_backtest_history(strategies: List[RSIReversalStrategy]):
    """현재 백테스트 결과 분석"""
    try:
        if not strategies:
            logger.warning("분석할 전략이 없습니다.")
            return
        results = []
        market_condition_results = {}
        for strategy in strategies:
            if not isinstance(strategy, RSIReversalStrategy):
                logger.warning(f"잘못된 전략 객체 타입: {type(strategy)}")
                continue
            try:
                metrics = strategy.calculate_metrics()
                if not isinstance(metrics, TradeMetrics):
                    logger.warning(f"잘못된 메트릭스 객체 타입: {type(metrics)}")
                    if isinstance(metrics, dict):
                        metrics = TradeMetrics(**metrics)
                    else:
                        continue
                win_rate = round(metrics.win_rate, 2)
                total_profit = round(metrics.total_profit, 2)
                avg_profit = round(metrics.avg_profit, 2)
                # market_condition이 Enum 객체면 value로 변환
                market_condition = metrics.market_condition
                if hasattr(market_condition, 'value'):
                    market_condition = market_condition.value
                results.append({
                    'ticker': strategy.ticker,
                    'total_trades': metrics.trade_count,
                    'win_rate': win_rate,
                    'total_profit': total_profit,
                    'avg_profit': avg_profit,
                    'market_condition': market_condition
                })
                market_state = market_condition
                if market_state not in market_condition_results:
                    market_condition_results[market_state] = {
                        'total_trades': [],
                        'win_rate': [],
                        'total_profit': [],
                        'avg_profit': []
                    }
                market_condition_results[market_state]['total_trades'].append(metrics.trade_count)
                market_condition_results[market_state]['win_rate'].append(win_rate)
                market_condition_results[market_state]['total_profit'].append(total_profit)
                market_condition_results[market_state]['avg_profit'].append(avg_profit)
            except Exception as e:
                logger.error(f"{strategy.ticker} 분석 중 오류: {str(e)}")
                continue
        if not results:
            logger.warning("분석할 결과가 없습니다.")
            return
        df = pd.DataFrame(results)
        print("\n" + "="*100)
        print("코인별 성과")
        print("-"*100)
        print(df[['ticker', 'total_trades', 'win_rate', 'total_profit', 'avg_profit']].to_string(index=False))
        print("="*100)
        market_stats = []
        for state, stats in market_condition_results.items():
            if not stats['total_trades']:
                continue
            market_stats.append({
                'market_condition': state,
                'total_trades': round(np.mean(stats['total_trades']), 2),
                'win_rate': round(np.mean(stats['win_rate']), 2),
                'total_profit': round(np.mean(stats['total_profit']), 2),
                'avg_profit': round(np.mean(stats['avg_profit']), 2)
            })
        if market_stats:
            df_market = pd.DataFrame(market_stats)
            df_market.set_index('market_condition', inplace=True)
            print("\n" + "="*100)
            print("시장상태별 성과")
            print("-"*100)
            print(df_market)
            print("="*100)
    except Exception as e:
        logger.error(f"결과 분석 중 오류: {str(e)}")
        return

def run_strategy_for_ticker(args):
    """단일 코인에 대한 전략 실행 (상세 디버깅)"""
    ticker, start_date_str, end_date_str, progress_dict, data_loading_progress = args
    
    try:
        # 로깅 레벨 임시 변경
        logger.setLevel(logging.WARNING)
        
        logger.info(f"\n{'='*80}\n{ticker} 전략 실행 시작\n{'='*80}")
        logger.info(f"백테스트 기간: {start_date_str} ~ {end_date_str}")
        
        # 데이터 로드
        logger.info(f"{ticker} 데이터 로드 중...")
        df = fetch_ohlcv(ticker, start_date_str, end_date_str)
        
        if df is None:
            logger.error(f"{ticker}: 데이터 로드 실패")
            return None
            
        logger.info(f"{ticker}: 데이터 로드 완료")
        logger.info(f"- 데이터 포인트: {len(df):,}개")
        logger.info(f"- 기간: {df.index[0]} ~ {df.index[-1]}")
        logger.info(f"- 시작가: {df['open'].iloc[0]:,.0f}")
        logger.info(f"- 종료가: {df['close'].iloc[-1]:,.0f}")
        
        if len(df) < MIN_DATA_POINTS:
            logger.error(f"{ticker}: 데이터 부족 (필요: {MIN_DATA_POINTS}, 실제: {len(df)})")
            return None
        
        # 전략 객체 생성 및 초기화
        strategy = RSIReversalStrategy(ticker, is_backtest=True, randomize=True)
        
        # 기술적 지표 계산
        logger.info(f"{ticker}: 기술적 지표 계산 중...")
        
        # RSI 계산
        df['rsi'] = strategy.calculate_rsi(df)
        logger.info(f"- RSI: {df['rsi'].iloc[-1]:.1f} (마지막 값)")
        
        # 볼린저 밴드 계산
        bb_lower, bb_middle, bb_upper = strategy.calculate_bollinger_bands(df)
        df['bb_lower'] = bb_lower
        df['bb_middle'] = bb_middle
        df['bb_upper'] = bb_upper
        logger.info(f"- BB: {bb_lower.iloc[-1]:.0f} / {bb_middle.iloc[-1]:.0f} / {bb_upper.iloc[-1]:.0f} (하단/중앙/상단)")
        
        strategy.df = df
        
        # 진행 상황 초기화
        progress_dict[ticker] = {
            'progress': 0.0,
            'trades': 0,
            'profit': 0.0,
            'status': 'running'
        }
        
        # 데이터 처리
        window_size = strategy.bb_period
        total_rows = len(df)
        processed_count = 0
        
        logger.info(f"\n{ticker} 거래 처리 시작")
        logger.info(f"- 전체 데이터: {total_rows:,}개")
        logger.info(f"- 윈도우 크기: {window_size}")
        
        for i in range(window_size, total_rows):
            try:
                # 현재 윈도우 데이터
                window = df.iloc[max(0, i-window_size):i+1].copy()
                
                # 매 100번째 처리마다 상태 출력
                if i % 100 == 0:
                    current_price = window['close'].iloc[-1]
                    current_rsi = window['rsi'].iloc[-1]
                    logger.debug(f"{ticker} 처리 중 - {i}/{total_rows} | 가격: {current_price:,.0f} | RSI: {current_rsi:.1f}")
                
                # 거래 처리
                trades = strategy.process_single_window(window)
                if trades:
                    processed_count += 1
                    logger.info(f"{ticker} 거래 발생: {trades}")
                
                # 진행률 업데이트
                if i % max(1, total_rows // 100) == 0:
                    progress = (i / total_rows) * 100
                    metrics = strategy.summary.calculate_metrics()
                    
                    progress_dict[ticker] = {
                        'progress': round(progress, 1),
                        'trades': metrics.total_trades,
                        'profit': round(metrics.total_profit, 2),
                        'status': 'running'
                    }
                    
            except Exception as e:
                logger.error(f"{ticker} 처리 중 오류 (i={i}): {str(e)}")
                continue
        
        # 최종 상태 업데이트
        final_metrics = strategy.summary.calculate_metrics()
        progress_dict[ticker] = {
            'progress': 100.0,
            'trades': final_metrics.total_trades,
            'profit': round(final_metrics.total_profit, 2),
            'status': 'completed'
        }
        
        logger.info(f"\n{ticker} 처리 완료")
        logger.info(f"- 처리된 데이터: {processed_count:,}개")
        logger.info(f"- 총 거래: {final_metrics.total_trades:,}건")
        logger.info(f"- 총 수익: {final_metrics.total_profit:,.2f}%")
        logger.info(f"{'='*80}\n")
        
        return strategy
        
    except Exception as e:
        logger.error(f"{ticker} 처리 중 오류: {str(e)}")
        return None

def clear_and_print_progress(strategies, start_date, end_date, progress_dict=None):
    """화면을 지우고 모든 전략의 진행 상황을 출력"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # 백테스트 기간 출력
    print(f"\n백테스트 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}\n")
    print("-" * 80)
    
    if progress_dict:
        print("백테스트 진행 상황:")
        for ticker in sorted(progress_dict.keys()):
            try:
                data = progress_dict[ticker]
                progress = data.get('progress', 0)
                trades = data.get('trades', 0)
                profit = data.get('profit', 0)
                status = data.get('status', '')
                
                # 상태에 따른 표시
                status_indicator = ""
                if status == 'loading':
                    status_indicator = "⌛"
                elif status == 'running':
                    status_indicator = "▶"
                elif status == 'completed':
                    status_indicator = "✓"
                elif status == 'error':
                    status_indicator = "✗"
                elif status == 'failed':
                    status_indicator = "⚠"
                
                # 프로그레스 바 생성
                bar_width = 30
                filled = int(progress * bar_width / 100)
                bar = '=' * filled + '-' * (bar_width - filled)
                
                # 진행 상황 출력
                print(f"{status_indicator} {ticker:<8} [{bar}] {progress:5.1f}% | 거래: {trades:4d}건 | 수익: {profit:+7.2f}%")
            except Exception as e:
                print(f"{ticker:<10} - 상태 업데이트 오류")
        
        print("-" * 80)
    else:
        print("진행 중인 백테스트가 없습니다.")
    
    print("-" * 80)

def save_backtest_results(strategies: List[RSIReversalStrategy]) -> str:
    """백테스트 결과를 JSON 파일로 저장"""
    try:
        # 결과 저장 디렉토리 생성
        results_dir = Path(__file__).parent / 'backtest_results'
        results_dir.mkdir(exist_ok=True)
        
        # 결과 파일명 생성
        filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = results_dir / filename
        
        # 결과 데이터 생성
        results_data = []
        for strategy in strategies:
            if not strategy:
                continue
            metrics = strategy.summary.calculate_metrics()
            if isinstance(metrics, dict):
                metrics = TradeMetrics(**metrics)
            results_data.append({
                'ticker': strategy.ticker,
                'total_trades': metrics.trade_count,
                'win_rate': round(metrics.win_rate, 2),
                'total_profit': round(metrics.total_profit, 2),
                'avg_profit': round(metrics.avg_profit, 2),
                'market_condition': metrics.market_condition,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        # JSON 파일로 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        return filepath
    except Exception as e:
        logger.error(f"결과 저장 중 오류: {str(e)}")
        return None

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # 숫자형 컬럼 변환
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        # NaN 처리
        df = df.fillna(method='ffill')
        # 무한값 처리
        df = df.replace([np.inf, -np.inf], np.nan)
        return df
    except Exception as e:
        logger.error(f"데이터 전처리 중 오류: {str(e)}")
        return df

def check_market_conditions(self, df: pd.DataFrame) -> Tuple[bool, str]:
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.error(f"잘못된 데이터 타입 또는 빈 데이터프레임: {type(df)}, empty={df.empty if isinstance(df, pd.DataFrame) else 'N/A'}")
            return False, "데이터 타입 오류"
        required_columns = ['close', 'high', 'low']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"필수 컬럼 누락: {[col for col in required_columns if col not in df.columns]}")
            return False, "컬럼 누락"
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[required_columns].isna().any().any():
            logger.warning("NaN 값 발견됨")
            df = df.fillna(method='ffill')
        # 마지막 값 안전 변환 (Series/배열/2차원 구조까지 방어)
        def safe_float(val):
            try:
                # Series of Series or ndarray of ndarray → 1차원으로 평탄화
                if isinstance(val, (pd.Series, np.ndarray)):
                    arr = np.array(val).flatten()
                    if arr.size > 0:
                        val = arr[-1]
                    else:
                        return 0.0
                if pd.isna(val) or val is None:
                    return 0.0
                return float(val)
            except Exception as e:
                logger.error(f'safe_float 변환 실패: {val}, type={type(val)}, 오류: {e}')
                return 0.0
        try:
            last_close = df['close'].iloc[-1]
            logger.debug(f'last_close iloc[-1] type: {type(last_close)}, value: {last_close}')
            current_price = safe_float(last_close)
            current_high = safe_float(df['high'].iloc[-1])
            current_low = safe_float(df['low'].iloc[-1])
        except Exception as e:
            logger.error(f'iloc[-1] 접근 실패: {e}')
            return False, '데이터 인덱스 오류'
        # (여기에 기존 시장 상태 체크 로직)
        return True, '정상'
    except Exception as e:
        logger.error(f'시장 상태 체크 중 오류: {str(e)}')
        return False, '오류 발생'

# safe_float 함수: 어디서든 사용할 수 있도록 전역에 정의
def safe_float(val):
    try:
        if isinstance(val, (pd.Series, np.ndarray)):
            arr = np.array(val).flatten()
            if arr.size > 0:
                val = arr[-1]
            else:
                return 0.0
        if pd.isna(val) or val is None:
            return 0.0
        return float(val)
    except Exception as e:
        print(f'safe_float 변환 실패: {val}, type={type(val)}, 오류: {e}')
        return 0.0

# 거래 로그 파일 경로 추가
TRADE_LOG_PATH = Path(__file__).parent / 'trade_log.txt'

# 기존 print/logging을 파일로 저장하는 함수 추가
def log_trade_to_file(trade_info):
    try:
        with open(TRADE_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(trade_info + '\n')
    except Exception as e:
        logger.error(f"거래 로그 파일 저장 실패: {e}")

if __name__ == "__main__":
    try:
        # 테스트할 티커 목록
        tickers = [
            "KRW-BTC",  # 비트코인
            "KRW-ETH",  # 이더리움
            "KRW-XRP",  # 리플
            "KRW-SOL",  # 솔라나
            "KRW-ADA",  # 에이다
            "KRW-DOGE", # 도지코인
        ]
        
        # 백테스트 기간 설정
        test_periods = [30, 60, 90, 180]
        selected_period = random.choice(test_periods)
        
        current_date = datetime.now()
        three_years_ago = current_date - timedelta(days=365*3)
        
        max_start_date = current_date - timedelta(days=selected_period)
        random_days = random.randint(0, (max_start_date - three_years_ago).days)
        start_date = three_years_ago + timedelta(days=random_days)
        end_date = start_date + timedelta(days=selected_period)
        
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"{LogEmoji.INFO} 백테스트 시작")
        logger.info(f"대상 티커: {', '.join(tickers)}")
        logger.info(f"테스트 기간: {selected_period}일 ({start_date_str} ~ {end_date_str})")
        
        # 멀티프로세싱 설정
        cpu_count = mp.cpu_count()
        process_count = min(len(tickers), cpu_count)
        logger.info(f"CPU 코어 수: {cpu_count}, 사용할 프로세스 수: {process_count}")
        
        # 공유 메모리 딕셔너리 생성
        with mp.Manager() as manager:
            try:
                progress_dict = manager.dict()
                active_strategies = manager.list()
                data_loading_progress = manager.dict()
                for ticker in tickers:
                    data_loading_progress[ticker] = {'loaded': False}
                for ticker in tickers:
                    strategy = RSIReversalStrategy(ticker, is_backtest=True, randomize=True)
                    strategy._progress_dict = {}
                    strategy._progress_dict[ticker] = {
                        'progress': 0.0,
                        'trades': 0,
                        'profit': 0.0,
                        'remaining_time': 0
                    }
                    active_strategies.append(strategy)
                tasks = [(ticker, start_date_str, end_date_str, progress_dict, data_loading_progress) for ticker in tickers]
                with ProcessPoolExecutor(max_workers=process_count) as executor:
                    futures = [executor.submit(run_strategy_for_ticker, task) for task in tasks]
                    try:
                        while futures:
                            done, not_done = concurrent.futures.wait(futures, timeout=0.5)
                            for future in done:
                                strategy = future.result()
                                if strategy is not None:
                                    for i, s in enumerate(active_strategies):
                                        if s.ticker == strategy.ticker:
                                            active_strategies[i] = strategy
                                            break
                                futures.remove(future)
                            clear_and_print_progress(list(active_strategies), start_date, end_date, progress_dict)
                            if not futures:
                                break
                            time.sleep(0.5)
                    except KeyboardInterrupt:
                        print("사용자 강제 종료 감지! 모든 작업을 취소합니다.")
                        for future in futures:
                            future.cancel()
                        executor.shutdown(wait=True, cancel_futures=True)
                        print("모든 작업이 안전하게 취소되었습니다.")
                    finally:
                        # 혹시 남아있는 future가 있으면 모두 취소
                        for future in futures:
                            if not future.done():
                                future.cancel()
                        executor.shutdown(wait=True, cancel_futures=True)
                print("\n" + "=" * 80)
                final_strategies = list(active_strategies)
                if final_strategies:
                    compare_strategies(final_strategies)
                    save_backtest_results(final_strategies)
                    analyze_backtest_history(final_strategies)
                logger.info(f"\n{LogEmoji.SUCCESS} 백테스트 완료")
            except KeyboardInterrupt:
                logger.warning("\n프로그램이 사용자에 의해 중단되었습니다.")
            except Exception as e:
                logger.error(f"\n프로그램 실행 중 오류: {str(e)}")
        print("모든 백테스트 및 결과 분석이 완료되었습니다.")
    except Exception as e:
        logger.error(f"{LogEmoji.ERROR} 프로그램 실행 중 오류: {str(e)}")
    finally:
        print("프로그램이 종료됩니다.")

# ====== [테스트: unknown_market_debug.log 생성 확인용] ======
import pandas as pd
test_df = pd.DataFrame({
    'close': [1]*59,
    'high': [1]*59,
    'low': [1]*59,
    'volume': [1]*59
})
test_strategy = RSIReversalStrategy("KRW-BTC")
result = test_strategy.check_market_conditions(test_df)
print(f"[테스트] check_market_conditions 결과: {result}")
# ... existing code ...