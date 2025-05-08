import os
import time
import logging
import pyupbit
import pandas as pd
from datetime import datetime, timedelta
import pickle
import glob
from pathlib import Path
import hashlib
from typing import Optional

# 로깅 설정
logger = logging.getLogger(__name__)

# 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_MAX_AGE_DAYS = 7

# 로그 이모지 정의
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

def print_progress(current: int, total: int, prefix: str = '', suffix: str = '', decimals: int = 1, length: int = 50, fill: str = '█', print_end: str = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    if current == total:
        print()

def safe_request_with_retry(func, *args, max_retries: int = 5, delay: float = 1.0, **kwargs):
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"요청 실패 (시도 {attempt + 1}/{max_retries}): {str(e)}")
        time.sleep(delay * (2 ** attempt))
    return None

def get_cache_key(ticker: str, start_date: str, end_date: str) -> str:
    """캐시 키 생성 함수"""
    try:
        # 날짜를 년월 형식으로 변환 (일단위가 아닌 월단위로 관리)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m")
        
        # 캐시 키 형식: ticker_YYYYMM_YYYYMM_interval
        key = f"{ticker}_{start_dt}_{end_dt}_minute5"
        
        # 해시 생성 (16자)
        cache_key = hashlib.md5(key.encode()).hexdigest()[:16]
        logger.debug(f"캐시 키 생성: {key} -> {cache_key}")
        return cache_key
    except Exception as e:
        logger.error(f"캐시 키 생성 실패: {str(e)}")
        return hashlib.md5(f"{ticker}_{start_date}_{end_date}".encode()).hexdigest()[:16]

def find_matching_cache(ticker: str, start_date: str, end_date: str):
    """비슷한 기간의 캐시된 데이터 찾기"""
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # 시간 정보 제거
        start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # 허용 오차 범위 확대 (7일 -> 30일)
        allowed_diff = timedelta(days=30)
        cache_files = glob.glob(os.path.join(CACHE_DIR, "*.pkl"))
        
        best_match = None
        best_coverage = 0
        
        for cache_file in cache_files:
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                    if not isinstance(data, pd.DataFrame) or data.empty:
                        continue
                    
                    # 데이터의 시작/종료일 확인
                    data_start = data.index[0].replace(hour=0, minute=0, second=0, microsecond=0)
                    data_end = data.index[-1].replace(hour=0, minute=0, second=0, microsecond=0)
                    
                    # 데이터 기간이 너무 짧으면 스킵 (최소 90일)
                    if (data_end - data_start).days < 90:
                        continue
                    
                    # 요청 기간을 포함하는지 확인
                    if data_start <= start_dt + allowed_diff and data_end >= end_dt - allowed_diff:
                        coverage = (min(data_end, end_dt) - max(data_start, start_dt)).days
                        
                        if coverage > best_coverage:
                            best_coverage = coverage
                            best_match = data
                            logger.info(f"더 나은 캐시 매치 발견: {os.path.basename(cache_file)}")
                            logger.info(f"  - 커버리지: {coverage}일")
                            logger.info(f"  - 기간: {data_start.date()} ~ {data_end.date()}")
                
            except Exception as e:
                logger.warning(f"캐시 파일 처리 중 오류: {str(e)}")
                continue
        
        if best_match is not None:
            # 데이터 필터링 시 여유 기간 포함
            mask = (best_match.index >= start_dt - allowed_diff) & (best_match.index <= end_dt + allowed_diff)
            filtered_data = best_match[mask].copy()
            
            if not filtered_data.empty and len(filtered_data) >= 1000:  # 최소 데이터 포인트 확인
                return filtered_data
        
        return None
        
    except Exception as e:
        logger.error(f"캐시 매칭 중 오류: {str(e)}")
        return None

def get_cached_data(cache_key: str):
    try:
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, pd.DataFrame) and not data.empty:
                    return data
        return None
    except Exception as e:
        logger.error(f"캐시 데이터 로드 중 오류: {str(e)}")
        return None

def clean_old_cache(cache_dir: Path = CACHE_DIR, max_age_days: int = CACHE_MAX_AGE_DAYS):
    """오래된 캐시 파일 정리"""
    try:
        now = datetime.now().timestamp()
        for cache_file in cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    if (now - data['timestamp']) > max_age_days * 86400:
                        cache_file.unlink()
            except:
                cache_file.unlink()  # 손상된 캐시 파일 삭제
                
    except Exception as e:
        logger.error(f"캐시 정리 중 오류: {str(e)}")

def save_to_cache(df: pd.DataFrame, cache_key: str) -> bool:
    """데이터를 캐시에 저장"""
    try:
        if df.empty or len(df) < 1000:  # 최소 데이터 포인트 확인
            logger.warning("캐시 저장 실패: 데이터가 비어있거나 너무 적습니다")
            return False
            
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
        
        # 기존 캐시 파일 백업
        if os.path.exists(cache_file):
            backup_file = cache_file + ".bak"
            try:
                os.rename(cache_file, backup_file)
            except Exception as e:
                logger.warning(f"캐시 파일 백업 실패: {str(e)}")
        
        # 새 데이터 저장
        with open(cache_file, "wb") as f:
            pickle.dump(df, f)
            
        logger.info(f"캐시 저장 완료: {len(df):,}개 데이터 포인트")
        
        # 백업 파일 삭제
        if os.path.exists(backup_file):
            try:
                os.remove(backup_file)
            except Exception as e:
                logger.warning(f"백업 파일 삭제 실패: {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"캐시 저장 중 오류: {str(e)}")
        return False

def fetch_ohlcv(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """OHLCV 데이터 조회 (캐시 시스템 포함)"""
    try:
        cache_key = get_cache_key(ticker, start_date, end_date)
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        
        # 캐시 확인
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                if (datetime.now().timestamp() - data['timestamp']) < CACHE_MAX_AGE_DAYS * 86400:
                    return data['data']
        
        # 새로운 데이터 조회
        df = pyupbit.get_ohlcv(ticker, interval="minute5", to=end_date, count=1000)
        if df is not None and len(df) > 0:
            # 캐시 저장
            cache_data = {
                'ticker': ticker,
                'data': df,
                'timestamp': datetime.now().timestamp()
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            return df
            
        return None
        
    except Exception as e:
        logging.error(f"데이터 조회 중 오류 ({ticker}): {str(e)}")
        return None 