# RSI 역추세 전략 자동 백테스트 (다중 알트코인)
# 주요 구성: 전략 클래스 + 다중 티커 루프 + 성과 비교 분석

import os
import time
import logging
import pyupbit
import pandas as pd
import numpy as np
import random  # random 모듈 임포트
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp
from pathlib import Path
import pickle
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# 로깅 설정
log_level = logging.INFO # 로그 레벨 설정 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
log_format = '%(asctime)s [%(levelname)s] %(processName)s:%(threadName)s - %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'

# 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"

# 디렉토리 생성
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log_file_path = LOG_DIR / f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    print(f"로그 파일 경로: {log_file_path}") # 경로 확인용 출력
except Exception as e:
    print(f"[ERROR] 디렉토리 생성 또는 로그 파일 경로 설정 중 오류: {e}")
    # 대체 경로 사용 (사용자 홈 디렉토리)
    user_home = Path.home()
    LOG_DIR = user_home / "volatility_bot_logs"
    CACHE_DIR = user_home / "volatility_bot_cache"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log_file_path = LOG_DIR / f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    print(f"대체 로그 파일 경로: {log_file_path}")

# 로거 설정
logger = logging.getLogger()
logger.setLevel(log_level)

# 기존 핸들러 제거 (선택적: 필요한 경우에만 사용)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# 포맷터 생성
formatter = logging.Formatter(log_format, datefmt=date_format)

# 스트림 핸들러 (콘솔 출력)
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 파일 핸들러 (파일 출력)
try:
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(log_level) # 파일 핸들러 레벨 명시적 설정
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("로깅 설정 완료. 콘솔 및 파일 로깅 시작.")
except Exception as e:
    print(f"[ERROR] 파일 핸들러 설정 중 오류: {e}")
    logger.error(f"파일 핸들러 설정 실패: {e}", exc_info=True)

# 디버그 로깅 비활성화 (외부 라이브러리)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

def check_ticker_validity(ticker: str) -> bool:
    """티커의 유효성을 검사합니다."""
    try:
        # 현재가 조회로 유효성 체크
        current_price = pyupbit.get_current_price(ticker)
        return current_price is not None
    except Exception as e:
        logging.error(f"티커 유효성 검사 실패 ({ticker}): {str(e)}")
        return False

def safe_request_with_retry(func, *args, max_retries: int = 5, delay: float = 1.0, **kwargs) -> Optional[pd.DataFrame]:
    """안전한 API 요청 처리를 위한 래퍼 함수"""
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            if result is not None:
                return result
        except Exception as e:
            logging.warning(f"요청 실패 (시도 {attempt + 1}/{max_retries}): {str(e)}")
        
        # 지수 백오프
        time.sleep(delay * (2 ** attempt))
    return None

class RSIReversalStrategy:
    def __init__(self, ticker: str, is_backtest: bool = False):
        self.ticker = ticker
        self.is_backtest = is_backtest
        
        # 기본 파라미터
        self.rsi_period = 14
        self.bb_period = 20
        self.bb_std = 2.0
        self.ma_short = 10  # 단기 이동평균
        self.ma_long = 30   # 장기 이동평균
        self.vol_window = 20  # 변동성 계산 기간
        
        # 손익 관련 파라미터
        self.take_profit = 0.08     # 8%로 상향 조정
        self.stop_loss = -0.04      # -4%로 하향 조정
        self.trailing_stop = 0.03   # 3%로 상향 조정
        self.max_hold_time = 96     # 최대 보유 시간 4일로 확대
        
        # 랜덤화 범위 조정
        if self.is_backtest:
            self.rsi_period = random.randint(12, 16)
            self.bb_period = random.randint(18, 22)
            self.bb_std = random.uniform(1.8, 2.2)
            self.ma_short = random.randint(8, 12)
            self.ma_long = random.randint(28, 32)
            
            # 손익 파라미터 랜덤화 (더 넓은 범위)
            self.take_profit = random.uniform(0.06, 0.10)
            self.stop_loss = random.uniform(-0.05, -0.03)
            self.trailing_stop = random.uniform(0.02, 0.04)
        
        # 거래 기록
        self.trade_history = []
        self.last_trade_time = None
        
    def calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """추세 강도 계산"""
        close_prices = df['close'].astype(float)
        ma_short = close_prices.rolling(window=self.ma_short).mean()
        ma_long = close_prices.rolling(window=self.ma_long).mean()
        
        # 추세 강도 계산 (단기/장기 이평선 비율)
        trend_strength = (ma_short.iloc[-1] / ma_long.iloc[-1] - 1) * 100
        return float(trend_strength)

    def calculate_volatility(self, df: pd.DataFrame) -> Tuple[float, float]:
        """변동성 계산"""
        high_prices = df['high'].astype(float)
        low_prices = df['low'].astype(float)
        
        # 일중 변동성
        daily_volatility = ((high_prices / low_prices - 1) * 100).rolling(window=self.vol_window).mean()
        
        # ATR 스타일 변동성
        tr = pd.DataFrame(index=df.index)
        tr['hl'] = high_prices - low_prices
        tr['hc'] = abs(high_prices - df['close'].shift(1))
        tr['lc'] = abs(low_prices - df['close'].shift(1))
        atr = tr.max(axis=1).rolling(window=self.vol_window).mean()
        
        return float(daily_volatility.iloc[-1]), float(atr.iloc[-1])

    def evaluate_entry(self, df: pd.DataFrame) -> bool:
        """진입 조건을 점수화하여 평가 (더 엄격한 조건)"""
        if len(df) < max(self.rsi_period, self.bb_period, self.ma_long) + 10:
            return False
        
        score = 0
        current_price = float(df['close'].iloc[-1])
        
        # RSI 조건 (0-3점)
        rsi = self.calculate_rsi(df)
        current_rsi = float(rsi.iloc[-1])
        if current_rsi < 25:  # 더 낮은 RSI 기준
            score += 3
        elif current_rsi < 30:
            score += 2
        elif current_rsi < 35:
            score += 1
        
        # RSI 하락 추세 확인
        rsi_values = rsi.iloc[-3:]
        if all(rsi_values.iloc[i] > rsi_values.iloc[i+1] for i in range(len(rsi_values)-1)):
            score += 2  # 연속 하락시 추가 점수
        
        # 볼린저 밴드 조건 (0-3점)
        bb_lower, bb_middle, bb_upper = self.calculate_bollinger_bands(df)
        bb_lower_val = float(bb_lower.iloc[-1])
        bb_middle_val = float(bb_middle.iloc[-1])
        
        if current_price < bb_lower_val:
            score += 3
        elif current_price < (bb_lower_val + bb_middle_val) / 2:
            score += 1
        
        # 거래량 조건 (0-3점)
        volume_ma = df['volume'].rolling(window=20).mean()
        current_volume = float(df['volume'].iloc[-1])
        volume_ma_val = float(volume_ma.iloc[-1])
        
        if current_volume > volume_ma_val * 2.0:  # 거래량 급증
            score += 3
        elif current_volume > volume_ma_val * 1.5:
            score += 2
        elif current_volume > volume_ma_val * 1.2:
            score += 1
        
        # 추세 강도 체크 (-2~2점)
        trend_strength = self.calculate_trend_strength(df)
        if trend_strength < -2.0:  # 강한 하락 추세
            score += 2
        elif trend_strength < -1.0:  # 약한 하락 추세
            score += 1
        elif trend_strength > 1.0:  # 상승 추세에서는 진입 제한
            score -= 2
        
        # 변동성 필터
        daily_vol, atr = self.calculate_volatility(df)
        if daily_vol > 5.0 or atr > 0.05:  # 변동성이 너무 높으면 진입 제한
            score -= 2
        
        # 추가 필터
        if self.last_trade_time and (df.index[-1] - self.last_trade_time).total_seconds() < 14400:  # 4시간으로 확대
            return False
        
        # 최소 8점 이상 획득시 진입 (더 높은 기준)
        return score >= 8
        
    def evaluate_exit(self, df: pd.DataFrame, entry_price: float, hold_time: float) -> Tuple[bool, str]:
        """퇴출 조건을 점수화하여 평가"""
        try:
            if len(df) < max(self.rsi_period, self.bb_period) + 10:
                return False, ""
            
            current_price = float(df['close'].iloc[-1])
            entry_price = float(entry_price)  # 명시적으로 float로 변환
            profit_ratio = (current_price / entry_price) - 1.0
            
            # 손절/이익실현
            if profit_ratio <= self.stop_loss:
                return True, "손절"
            if profit_ratio >= self.take_profit:
                return True, "이익실현"
            
            score = 0
            
            # RSI 조건 (0-2점)
            rsi = self.calculate_rsi(df)
            current_rsi = float(rsi.iloc[-1])
            if current_rsi > 65:  # 과매수
                score += 2
            elif current_rsi > 60:  # 약한 과매수
                score += 1
            
            # 볼린저 밴드 조건 (0-2점)
            lower, middle, upper = self.calculate_bollinger_bands(df)
            current_price = float(df['close'].iloc[-1])
            bb_upper = float(upper.iloc[-1])
            bb_middle = float(middle.iloc[-1])
            
            if current_price > bb_upper:  # 상단 밴드 위
                score += 2
            elif current_price > (bb_upper + bb_middle) / 2:  # 상단과 중간 사이
                score += 1
            
            # 보유 시간 가중치
            if hold_time >= self.max_hold_time * 0.8:  # 최대 보유 시간의 80% 이상
                score += 2
            
            # 트레일링 스탑
            if profit_ratio > self.take_profit * 0.5:
                high_since_entry = float(df['high'].max())  # 전체 최고가 사용
                trailing_ratio = (current_price / high_since_entry) - 1.0
                if trailing_ratio <= -self.trailing_stop:
                    return True, "트레일링 스탑"
                
            # 최소 3점 이상 획득시 퇴출
            if score >= 3:
                return True, "조건 충족"
            
            return False, ""
        
        except Exception as e:
            logging.error(f"퇴출 평가 중 오류 발생: {str(e)}")
            return False, ""

    def calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """RSI 계산 시 0으로 나누는 문제 방지"""
        # 데이터 타입을 float으로 변환
        close_prices = df['close'].astype(float)
        delta = close_prices.diff()
        
        # 상승폭과 하락폭 계산
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        # 평균 계산
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        # 0으로 나누는 것을 방지
        avg_loss = avg_loss.replace(0, float('inf'))
        rs = avg_gain / avg_loss
        
        # RSI 계산
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)  # NaN 값을 50으로 대체

    def calculate_bollinger_bands(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """볼린저 밴드 계산"""
        # 데이터 타입을 float으로 변환
        close_prices = df['close'].astype(float)
        
        # 중간 밴드 (20일 이동평균)
        middle = close_prices.rolling(window=self.bb_period).mean()
        
        # 표준편차 계산
        std = close_prices.rolling(window=self.bb_period).std()
        
        # 상단과 하단 밴드
        upper = middle + (std * float(self.bb_std))
        lower = middle - (std * float(self.bb_std))
        
        # NaN 값 처리
        middle = middle.fillna(close_prices)
        upper = upper.fillna(close_prices * 1.02)  # NaN인 경우 현재가 + 2%
        lower = lower.fillna(close_prices * 0.98)  # NaN인 경우 현재가 - 2%
        
        return lower, middle, upper

def get_cache_key(ticker: str, start_date: str, end_date: str) -> str:
    """캐시 키 생성 (개선된 버전)"""
    key = f"{ticker}_{start_date}_{end_date}_minute5"
    cache_key = hashlib.md5(key.encode()).hexdigest()
    logging.debug(f"캐시 키 생성: {key} -> {cache_key}")
    return cache_key

def get_cached_data(cache_key: str, cache_dir: str = CACHE_DIR) -> Optional[pd.DataFrame]:
    """캐시된 데이터 가져오기 (개선된 버전)"""
    try:
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
        logging.debug(f"캐시 파일 확인: {cache_file}")
        
        if os.path.exists(cache_file):
            # 캐시 파일의 수정 시간 확인
            cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            cache_age = datetime.now() - cache_mtime
            
            # 캐시가 24시간 이상 지났으면 무효화
            if cache_age.total_seconds() > 86400:  # 24시간
                logging.info(f"캐시 파일 만료: {cache_file}")
                os.remove(cache_file)  # 만료된 캐시 파일 삭제
                return None
                
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                
            if isinstance(data, pd.DataFrame) and not data.empty:
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                if all(col in data.columns for col in required_columns):
                    logging.info(f"캐시 데이터 로드 성공: {cache_file} (행: {len(data)})")
                    return data
                    
        else:
            logging.debug(f"캐시 파일 없음: {cache_file}")
                    
    except Exception as e:
        logging.warning(f"캐시 파일 로드 실패: {str(e)}")
    return None

def save_to_cache(data: pd.DataFrame, cache_key: str, cache_dir: str = CACHE_DIR) -> None:
    """데이터를 캐시에 저장 (개선된 버전)"""
    try:
        if data is None or data.empty:
            logging.warning("빈 데이터는 캐시하지 않습니다")
            return
            
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
        logging.debug(f"캐시 저장 시도: {cache_file}")
        
        # 캐시 디렉토리가 없으면 생성
        os.makedirs(cache_dir, exist_ok=True)
        
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        logging.info(f"캐시 저장 성공: {cache_file} (행: {len(data)})")
        
    except Exception as e:
        logging.warning(f"캐시 저장 실패: {str(e)}")

def fetch_ohlcv(ticker: str, start_date: str, end_date: str, interval: str = "minute5", allow_partial: bool = True) -> pd.DataFrame:
    """OHLCV 데이터를 가져오는 함수"""
    cache_key = get_cache_key(ticker, start_date, end_date)
    cached_data = get_cached_data(cache_key)
    
    if cached_data is not None:
        print(f"{ticker}: 캐시된 데이터 사용")
        print(f"데이터 기간: {cached_data.index[0].strftime('%Y-%m-%d')} ~ {cached_data.index[-1].strftime('%Y-%m-%d')}")
        return cached_data

    print(f"\n{ticker}: 새로운 데이터 수집 시작")
    print(f"요청 기간: {start_date} ~ {end_date}")
    
    # 날짜 변환
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    all_data = []
    current_dt = end_dt
    total_days = (end_dt - start_dt).days
    days_processed = 0
    last_progress = -1
    last_update_time = datetime.now()
    retries = 0
    max_retries = 3
    no_progress_timeout = 300  # 5분 동안 진행이 없으면 타임아웃
    collection_status = "완료"  # 수집 상태 추적
    
    def print_progress(force=False):
        nonlocal last_progress, last_update_time
        current_time = datetime.now()
        elapsed = (current_time - last_update_time).total_seconds()
        progress = min(100, int((days_processed / total_days) * 100))
        
        if progress != last_progress or force:
            time_str = current_time.strftime("%H:%M:%S")
            if elapsed > 60:
                status = f"(마지막 업데이트: {int(elapsed)}초 전)"
            else:
                status = "(진행중)"
            print(f"\r{ticker} 진행률: {progress}% - {time_str} {status}", end='')
            last_progress = progress
            if progress != last_progress:
                last_update_time = current_time
            
        return progress, elapsed
    
    while current_dt >= start_dt:
        try:
            df = safe_request_with_retry(
                lambda: pyupbit.get_ohlcv(ticker=ticker, interval=interval, count=200, to=current_dt),
                max_retries=3,
                delay=1.0
            )
            
            if df is None or df.empty:
                collection_status = "데이터 없음"
                logger.info(f"{ticker}: 더 이상의 과거 데이터가 없습니다. (현재: {current_dt.strftime('%Y-%m-%d')})")
                break
                
            all_data.append(df)
            
            # 진행률 업데이트
            current_dt = df.index[0]
            days_processed = (end_dt - current_dt).days
            progress, elapsed = print_progress()
            
            # 타임아웃 체크
            if elapsed > no_progress_timeout:
                collection_status = "시간 초과"
                logger.warning(f"{ticker}: {no_progress_timeout}초 동안 진행이 없어 수집을 중단합니다.")
                break
            
            if progress >= 100:
                break
                
            time.sleep(0.1)
            retries = 0
            
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                collection_status = "오류"
                logger.error(f"{ticker} 데이터 수집 중 최대 재시도 횟수 초과: {str(e)}")
                break
            logger.warning(f"{ticker} 데이터 수집 중 오류 (재시도 {retries}/{max_retries}): {str(e)}")
            time.sleep(retries * 2)
    
    # 마지막 진행률 강제 출력
    print_progress(force=True)
    print()  # 줄바꿈
    
    if not all_data:
        logger.warning(f"{ticker}: 수집된 데이터가 없습니다.")
        return pd.DataFrame()
    
    try:
        final_df = pd.concat(all_data).sort_index()
        final_df = final_df[~final_df.index.duplicated(keep='first')]
        
        if not final_df.empty:
            actual_start = final_df.index[0].strftime('%Y-%m-%d')
            actual_end = final_df.index[-1].strftime('%Y-%m-%d')
            completeness = (len(final_df) / (total_days * 288)) * 100  # 5분 봉 기준 하루 288개
            
            print(f"\n{ticker} 데이터 수집 결과:")
            print(f"- 상태: {collection_status}")
            print(f"- 실제 수집 기간: {actual_start} ~ {actual_end}")
            print(f"- 데이터 완성도: {completeness:.1f}%")
            print(f"- 총 데이터 포인트: {len(final_df):,}개")
            
            if collection_status != "완료" and not allow_partial:
                logger.warning(f"{ticker}: 부분 데이터 수집됨 (allow_partial=False)")
                return pd.DataFrame()
            
            save_to_cache(final_df, cache_key)
            logger.info(f"{ticker}: 데이터 저장 완료 (행: {len(final_df)})")
            
            return final_df
    except Exception as e:
        logger.error(f"{ticker} 데이터 처리 중 오류: {str(e)}")
        return pd.DataFrame()

def parallel_fetch_data(args) -> tuple:
    """병렬 처리를 위한 데이터 가져오기 함수 (캐시 활용)"""
    ticker, start_date, end_date = args
    
    # 프로세스별 로거 설정
    process_logger = logging.getLogger(f'Process-{ticker}')
    process_logger.setLevel(logging.DEBUG)
    
    if not process_logger.handlers:
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(formatter)
        process_logger.addHandler(console_handler)
    
    try:
        process_logger.info(f"{ticker}: 데이터 로드 시작")
        
        # 캐시 키 생성 및 확인
        cache_key = get_cache_key(ticker, start_date, end_date)
        cached_data = get_cached_data(cache_key)
        
        if cached_data is not None:
            process_logger.info(f"{ticker}: 캐시된 데이터 사용 (행: {len(cached_data)})")
            return ticker, cached_data
            
        process_logger.info(f"{ticker}: 새로운 데이터 수집 시작")
        df = fetch_ohlcv(ticker, start_date, end_date, allow_partial=True)
        
        if df.empty:
            process_logger.warning(f"{ticker}: 데이터 없음")
            return ticker, pd.DataFrame()
            
        # 새로 받은 데이터 캐시에 저장
        save_to_cache(df, cache_key)
        process_logger.info(f"{ticker}: 새로운 데이터 로드 완료 (행: {len(df)})")
        return ticker, df
        
    except Exception as e:
        process_logger.error(f"{ticker} 데이터 가져오기 실패: {str(e)}")
        return ticker, pd.DataFrame()

def backtest_strategy(strategy, df):
    """전략 백테스트 실행"""
    trades = []
    holding = False
    entry_price = None
    entry_time = None
    position_size = 1.0
    
    # DataFrame을 복사하여 사용
    df = df.copy()
    # 숫자형 컬럼을 float 타입으로 변환
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    for i in range(len(df)):
        current_time = df.index[i]
        current_price = float(df['close'].iloc[i])
        
        if not holding:
            # 진입 평가
            if i >= max(strategy.rsi_period, strategy.bb_period) + 10:
                window_df = df.iloc[max(0, i-100):i+1].copy()
                try:
                    if strategy.evaluate_entry(window_df):
                        holding = True
                        entry_price = current_price
                        entry_time = current_time
                        strategy.last_trade_time = current_time
                except Exception as e:
                    logging.error(f"진입 평가 중 오류 발생: {str(e)}")
                    continue
        else:
            # 보유 시간 계산
            hold_time = (current_time - entry_time).total_seconds() / 3600
            
            # 퇴출 평가
            window_df = df.iloc[max(0, i-100):i+1].copy()
            try:
                exit_flag, reason = strategy.evaluate_exit(window_df, entry_price, hold_time)
                
                if exit_flag:
                    profit_ratio = float(current_price) / float(entry_price) - 1
                    trade_result = {
                        "entry": entry_time,
                        "exit": current_time,
                        "pnl": float(profit_ratio * 100 * position_size),
                        "reason": reason,
                        "size": float(position_size)
                    }
                    trades.append(trade_result)
                    
                    holding = False
                    entry_price = None
                    entry_time = None
                    position_size = 1.0
            except Exception as e:
                logging.error(f"퇴출 평가 중 오류 발생: {str(e)}")
                continue

    return pd.DataFrame(trades)

def analyze_results(df_trades: pd.DataFrame) -> Dict:
    if df_trades.empty:
        return {
            "trades": 0,
            "win_rate": 0,
            "total_return": 0,
            "sharpe": 0
        }
    wins = df_trades[df_trades['pnl'] > 0]
    pnl_std = df_trades['pnl'].std()
    return {
        "trades": len(df_trades),
        "win_rate": len(wins) / len(df_trades) * 100,
        "total_return": df_trades['pnl'].sum(),
        "sharpe": np.sqrt(252) * df_trades['pnl'].mean() / pnl_std if pnl_std != 0 else 0
    }

def print_trade_summary(ticker: str, result: Dict) -> None:
    """거래 결과 요약을 예쁘게 출력"""
    print("\n" + "="*50)
    print(f"🪙 {ticker} 백테스트 결과")
    print("-"*50)
    print(f"📊 총 거래 횟수: {result['trades']}회")
    print(f"✨ 승률: {result['win_rate']:.2f}%")
    
    # 수익률에 따른 이모지 선택
    if result['total_return'] > 0:
        return_emoji = "🔥"
    elif result['total_return'] < 0:
        return_emoji = "📉"
    else:
        return_emoji = "➖"
    print(f"{return_emoji} 총 수익률: {result['total_return']:.2f}%")
    
    # 샤프비율에 따른 이모지 선택
    if result['sharpe'] > 1:
        sharpe_emoji = "⭐"
    elif result['sharpe'] > 0:
        sharpe_emoji = "✅"
    else:
        sharpe_emoji = "⚠️"
    print(f"{sharpe_emoji} 샤프비율: {result['sharpe']:.2f}")
    print("="*50)

def run_multi_coin_backtest(tickers: List[str], start_date: str, end_date: str, num_simulations: int = 1):
    """멀티 코인 백테스트 실행 (개선된 버전)"""
    # 데이터 로딩 진행률 표시 추가
    print("\n데이터 로딩 시작...")
    total_tickers = len(tickers)
    
    # 병렬 처리를 위한 프로세스 풀 생성
    num_processes = min(mp.cpu_count(), total_tickers)
    fetch_args = [(ticker, start_date, end_date) for ticker in tickers]
    
    all_data = {}
    with mp.Pool(num_processes) as pool:
        for i, (ticker, df) in enumerate(pool.imap_unordered(parallel_fetch_data, fetch_args), 1):
            print(f"\r데이터 로딩 진행률: {i}/{total_tickers} ({i/total_tickers*100:.1f}%)", end="")
            if not df.empty:
                all_data[ticker] = df
    
    print("\n데이터 로딩 완료!")
    print(f"로드된 코인 수: {len(all_data)}/{total_tickers}")
    
    # 이후 코드는 동일하게 유지
    all_results = []
    
    print("📈 백테스트 시작")
    print(f"📅 기간: {start_date} ~ {end_date}")
    print(f"🎯 대상 코인: {', '.join(tickers)}")
    print(f"🔄 시뮬레이션 횟수: {num_simulations}")

    for sim in range(num_simulations):
        print(f"--- 시뮬레이션 {sim + 1}/{num_simulations} ---")
        summary = {}
        total_tickers = len(tickers)

        for idx, ticker in enumerate(tickers, 1):
            try:
                print(f"⏳ 진행률: {idx}/{total_tickers} - {ticker} 분석 중 (시뮬레이션 {sim + 1})...")
                
                df = fetch_ohlcv(ticker, start_date, end_date)
                if df.empty:
                    logging.warning(f"{ticker}: 데이터 없음, 건너뜀")
                    continue
                
                # 코인별로 전략 인스턴스 생성 (시뮬레이션마다 새 파라미터 생성)
                strategy = RSIReversalStrategy(ticker)
                # 백테스트 실행 시 strategy 객체 전달 확인
                trades = backtest_strategy(strategy, df.copy()) # 원본 데이터 보존 위해 복사본 전달
                result = analyze_results(trades)
                summary[ticker] = result
                
                # 첫 시뮬레이션에서만 상세 로그 저장 (선택적)
                if sim == 0 and not trades.empty:
                     trades_filename = os.path.join(LOG_DIR, f"trades_{ticker}_{start_date}_{end_date}_sim{sim+1}.csv")
                     trades.to_csv(trades_filename)
                     print(f"📄 {ticker} 거래 내역 저장됨: {trades_filename}")

                print_trade_summary(ticker, result)
                
            except Exception as e:
                logging.error(f"{ticker} 백테스트 중 오류 발생 (시뮬레이션 {sim + 1}): {str(e)}")
                summary[ticker] = {
                    "trades": 0,
                    "win_rate": 0,
                    "total_return": 0,
                    "sharpe": 0
                }
        
        result_df = pd.DataFrame(summary).T
        result_df['simulation'] = sim + 1
        all_results.append(result_df)

    # 모든 시뮬레이션 결과 집계
    final_results_df = pd.concat(all_results)

    print("🎉 전체 백테스트 완료!")
    
    if not final_results_df.empty:
        # 시뮬레이션 전체 결과 요약
        print("=== 📊 전체 시뮬레이션 결과 요약 (평균값) ===")
        # numeric_only=True 추가하여 에러 방지
        mean_results = final_results_df.groupby(final_results_df.index).mean(numeric_only=True)
        print(f"📈 평균 승률: {mean_results['win_rate'].mean():.2f}%")
        print(f"💰 평균 수익률: {mean_results['total_return'].mean():.2f}%")
        print(f"📊 평균 샤프비율: {mean_results['sharpe'].mean():.2f}")
        print(f"🔄 평균 거래횟수 (코인당): {mean_results['trades'].mean():.1f}회")
        
        # 결과 저장
        results_filename = os.path.join(LOG_DIR, f"rsi_reversal_multi_result_simulations_{start_date}_{end_date}.csv")
        final_results_df.to_csv(results_filename)
        print(f"\n💾 전체 시뮬레이션 결과 저장 완료: {results_filename}")

        # 최고/최저 성과 코인 (평균 기준)
        best_coin_avg = mean_results['total_return'].idxmax()
        worst_coin_avg = mean_results['total_return'].idxmin()
        print(f"\n🏆 최고 성과 (평균): {best_coin_avg} ({mean_results.loc[best_coin_avg, 'total_return']:.2f}%)")
        print(f"\n⚠️ 최저 성과 (평균): {worst_coin_avg} ({mean_results.loc[worst_coin_avg, 'total_return']:.2f}%)")
    else:
        print("\n결과 데이터가 없어 요약을 생성할 수 없습니다.")

    return final_results_df # 최종 결과 반환

def analyze_market_condition(df: pd.DataFrame) -> dict:
    """시장 상태 분석"""
    # 변동성 계산
    returns = df['close'].pct_change()
    volatility = returns.std() * np.sqrt(252)  # 연간화된 변동성
    
    # 추세 강도 계산
    ma_short = df['close'].rolling(window=20).mean()  # 20일 이동평균
    ma_long = df['close'].rolling(window=60).mean()   # 60일 이동평균
    trend_strength = (ma_short.iloc[-1] / ma_long.iloc[-1] - 1) * 100
    
    # 거래량 특성
    volume_ma = df['volume'].rolling(window=20).mean()
    volume_ratio = df['volume'].iloc[-1] / volume_ma.iloc[-1]
    
    # 시장 상태 판단
    market_state = "횡보장"
    if trend_strength > 1.5 and volatility < 0.5:
        market_state = "강세장"
    elif trend_strength < -1.5 and volatility > 0.8:
        market_state = "약세장"
    elif volatility > 1.0:
        market_state = "고변동성"
    
    return {
        "volatility": volatility,
        "trend_strength": trend_strength,
        "volume_ratio": volume_ratio,
        "market_state": market_state
    }

def run_walk_forward_backtest(tickers: List[str], start_date: str, end_date: str,
                          lookback_days: int, test_days: int, num_simulations: int = 1):
    """Walk-Forward 백테스트 실행 (메타데이터 분석 추가)"""
    print("\n📈 Walk-Forward 백테스트 시작")
    print(f"📅 전체 기간: {start_date} ~ {end_date}")
    print(f"🗓️ 학습 기간: {lookback_days}일, 테스트 기간: {test_days}일")
    print(f"🎯 대상 코인: {', '.join(tickers)}")
    print(f"🔄 시뮬레이션 횟수 (창별): {num_simulations}")
    
    all_results = []
    window_results = []
    
    print("\n데이터 로딩 중...")
    data_dict = {}
    for ticker in tickers:
        print(f"\n{ticker} 데이터 로드 중...")
        df = fetch_ohlcv(ticker, start_date, end_date)
        if len(df) > 0:
            data_dict[ticker] = df
    
    print(f"\n데이터 로딩 완료. (로드된 코인: {len(data_dict)}개)\n")
    
    # 테스트 기간 설정
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    test_start = end_dt - timedelta(days=test_days)
    
    window_num = 1
    print(f"--- Window {window_num} --- ({test_start.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}) ---\n")
    
    for ticker in tickers:
        if ticker not in data_dict:
            continue
            
        df = data_dict[ticker]
        if len(df) == 0:
            continue
            
        # 시장 상태 분석
        market_condition = analyze_market_condition(df)
        print(f"\n{ticker} 시장 상태:")
        print(f"- 변동성: {market_condition['volatility']:.2f}")
        print(f"- 추세 강도: {market_condition['trend_strength']:.2f}")
        print(f"- 거래량 비율: {market_condition['volume_ratio']:.2f}")
        print(f"- 시장 국면: {market_condition['market_state']}")
            
        print(f"\n⏳ {ticker} (Window {window_num}) 분석 중 ({num_simulations}회 시뮬레이션)...")
        
        window_trades = []
        sim_results = []
        for sim in range(num_simulations):
            try:
                strategy = RSIReversalStrategy(ticker, is_backtest=True)
                test_data = df[df.index >= test_start]
                trades_df = backtest_strategy(strategy, test_data)
                
                if not trades_df.empty:
                    trades_df['simulation'] = sim + 1
                    window_trades.append(trades_df)
                    
                    # 시뮬레이션별 결과 저장
                    sim_result = analyze_results(trades_df)
                    sim_result.update({
                        'simulation': sim + 1,
                        'market_condition': market_condition['market_state']
                    })
                    sim_results.append(sim_result)
                    
            except Exception as e:
                logging.error(f"{ticker} (Window {window_num}, Sim {sim+1}) 백테스트 오류: {str(e)}")
                continue
        
        # 시뮬레이션 결과 분석
        if sim_results:
            sim_results_df = pd.DataFrame(sim_results)
            
            # 통계 분석
            stats = {
                'avg_return': sim_results_df['total_return'].mean(),
                'std_return': sim_results_df['total_return'].std(),
                'worst_return': sim_results_df['total_return'].min(),
                'best_return': sim_results_df['total_return'].max(),
                'avg_trades': sim_results_df['trades'].mean(),
                'avg_win_rate': sim_results_df['win_rate'].mean(),
                'market_state': market_condition['market_state']
            }
            
            print("\n==================================================")
            print(f" {ticker} (Window {window_num}) 통계 분석")
            print("--------------------------------------------------")
            print(f"📊 평균 거래 횟수: {stats['avg_trades']:.1f}회")
            print(f"✨ 평균 승률: {stats['avg_win_rate']:.2f}%")
            print(f"📈 평균 수익률: {stats['avg_return']:.2f}%")
            print(f"📊 수익률 표준편차: {stats['std_return']:.2f}%")
            print(f"⭐ 최고 수익률: {stats['best_return']:.2f}%")
            print(f"⚠️ 최저 수익률: {stats['worst_return']:.2f}%")
            print(f"🌍 시장 상태: {stats['market_state']}")
            print("==================================================\n")
            
            stats.update({
                'ticker': ticker,
                'window': window_num
            })
            window_results.append(stats)
            
    # 결과 저장
    results_df = pd.DataFrame(window_results)
    save_path = os.path.join(LOG_DIR, f"rsi_reversal_detailed_results_{start_date}_{end_date}.csv")
    results_df.to_csv(save_path, index=False)
    print(f"\n💾 상세 분석 결과 저장 완료: {save_path}")
    
    # 시장 상태별 성과 분석
    if not results_df.empty:
        print("\n=== 📊 시장 상태별 평균 성과 ===")
        market_stats = results_df.groupby('market_state').agg({
            'avg_return': 'mean',
            'worst_return': 'min',
            'avg_win_rate': 'mean',
            'avg_trades': 'mean'
        }).round(2)
        print(market_stats)
    
    return results_df

if __name__ == "__main__":
    logger.info("===== 프로그램 실행 시작 =====")
    try:
        # BTC를 첫번째로, DOGE를 두번째로 순서 변경
        altcoins = ["KRW-BTC", "KRW-DOGE", "KRW-XRP", "KRW-SOL", "KRW-SAND", "KRW-ARB"]
        
        # 3년치 데이터 로드 (ARB는 상장일 이후부터)
        current_date = datetime.now()
        three_years_ago = (current_date - timedelta(days=365*3)).strftime("%Y-%m-%d")
        current_date_str = current_date.strftime("%Y-%m-%d")
        
        print("\n3년치 데이터 로드 중...")
        all_data = {}
        for ticker in altcoins:
            print(f"{ticker} 데이터 로드 중...")
            # ARB의 경우 더 짧은 기간 설정 (2023년 3월 상장)
            start_date = "2023-03-01" if ticker == "KRW-ARB" else three_years_ago
            df = fetch_ohlcv(ticker, start_date, current_date_str, allow_partial=True)  # 부분 데이터 허용
            if not df.empty:
                all_data[ticker] = df
                print(f"{ticker}: {len(df)}개 데이터 포인트 로드됨")
            else:
                logger.warning(f"{ticker}: 데이터 로드 실패")
        
        # 여러 번의 랜덤 윈도우 테스트 실행
        num_windows = 5  # 랜덤 윈도우 횟수
        all_window_results = []
        
        for window in range(num_windows):
            print(f"\n=== 랜덤 윈도우 테스트 {window + 1}/{num_windows} ===")
            
            # 랜덤한 120일 구간 선택 (최근 2년 내에서)
            random_start_days = random.randint(120, 730)  # 최대 2년 전까지
            end_date = (current_date - timedelta(days=random_start_days)).strftime("%Y-%m-%d")
            start_date = (current_date - timedelta(days=random_start_days + 120)).strftime("%Y-%m-%d")
            
            print(f"\n선택된 백테스트 기간:")
            print(f"시작일: {start_date}")
            print(f"종료일: {end_date}")
            print(f"기간 길이: 120일\n")
            
            # 선택된 기간의 데이터만 필터링
            filtered_data = {}
            for ticker, df in all_data.items():
                mask = (df.index >= start_date) & (df.index <= end_date)
                filtered_df = df[mask].copy()
                if not filtered_df.empty:
                    filtered_data[ticker] = filtered_df
                    print(f"{ticker}: 선택된 기간 데이터 포인트 수 - {len(filtered_df)}")
            
            lookback = 60
            test = 30
            num_simulations = 5

            try:
                # Walk-Forward 백테스트 실행
                window_results = run_walk_forward_backtest(
                    list(filtered_data.keys()), start_date, end_date, lookback, test, num_simulations
                )
                all_window_results.append(window_results)
            except Exception as e:
                logging.error(f"윈도우 {window + 1} 실행 중 오류 발생: {str(e)}")
                continue
        
        # 전체 결과 통합 분석
        if all_window_results:
            combined_results = pd.concat(all_window_results, ignore_index=True)
            
            print("\n=== 📊 전체 테스트 결과 통계 ===")
            print("\n1. 전체 평균 성과:")
            print(f"평균 수익률: {combined_results['avg_return'].mean():.2f}% (표준편차: {combined_results['std_return'].mean():.2f}%)")
            print(f"평균 승률: {combined_results['avg_win_rate'].mean():.2f}%")
            print(f"평균 거래 횟수: {combined_results['avg_trades'].mean():.1f}")
            
            print("\n2. Worst-Case 시나리오:")
            print(f"최저 수익률: {combined_results['worst_return'].min():.2f}%")
            print(f"최저 승률: {combined_results['avg_win_rate'].min():.2f}%")
            
            print("\n3. 시장 상태별 성과:")
            market_performance = combined_results.groupby('market_state').agg({
                'avg_return': ['mean', 'std'],
                'worst_return': 'min',
                'avg_win_rate': 'mean',
                'avg_trades': 'mean'
            }).round(2)
            print(market_performance)
            
            # 최종 결과 저장
            final_save_path = os.path.join(LOG_DIR, f"rsi_reversal_complete_analysis_{current_date_str}.csv")
            combined_results.to_csv(final_save_path, index=False)
            print(f"\n💾 최종 분석 결과 저장 완료: {final_save_path}")

        logger.info("===== 모든 윈도우 테스트 및 분석 완료 ====")
        
    except Exception as e:
        logger.error(f"프로그램 실행 중 치명적인 오류 발생: {str(e)}", exc_info=True)
        import traceback
        traceback.print_exc()
    finally:
        logger.info("===== 프로그램 실행 종료 =====") # 종료 로그 추가
        logging.shutdown() # 로깅 시스템 종료 (버퍼 플러시)
