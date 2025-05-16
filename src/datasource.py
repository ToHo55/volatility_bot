import os
import time
from datetime import datetime, timedelta
import pandas as pd
import pyupbit
from loguru import logger
from typing import Optional, Dict, List
import requests
import numpy as np
import logging

class DataSource:
    def __init__(self, cache_dir: str = "data/raw"):
        """
        데이터 소스 초기화
        
        Args:
            cache_dir (str): 캐시 디렉토리 경로
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # API 호출 제한 설정
        self.upbit_rate_limit = 60  # 1분당 최대 호출 횟수
        self.last_call_time = 0
        self.call_count = 0
        self.rate_limit_reset_time = time.time() + 60  # 1분 후 리셋
        
        self.logger = logging.getLogger('trading_bot')
        self.logger.setLevel(logging.INFO)
        
        # 로그 디렉토리 생성
        os.makedirs('logs', exist_ok=True)
        
        # 파일 핸들러 설정 (UTF-8 인코딩 사용)
        file_handler = logging.FileHandler('logs/trading.log', encoding='utf-8')
        error_handler = logging.FileHandler('logs/error.log', encoding='utf-8')
        
        # 포맷터 설정
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        
        # 캐시 딕셔너리 초기화
        self._cache = {}
        
    def _rate_limit(self):
        """
        API 호출 제한 준수
        - 1분당 최대 60회 호출 제한
        - 호출 간격 최소 1초
        """
        current_time = time.time()
        
        # 1분이 지났으면 카운터 리셋
        if current_time >= self.rate_limit_reset_time:
            self.call_count = 0
            self.rate_limit_reset_time = current_time + 60
            
        # 호출 횟수 체크
        if self.call_count >= self.upbit_rate_limit:
            wait_time = self.rate_limit_reset_time - current_time
            if wait_time > 0:
                logger.warning(f"API 호출 제한 도달. {wait_time:.1f}초 대기")
                time.sleep(wait_time)
                self.call_count = 0
                self.rate_limit_reset_time = time.time() + 60
                
        # 호출 간격 체크
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < 1.0:  # 최소 1초 간격
            time.sleep(1.0 - time_since_last_call)
            
        self.last_call_time = time.time()
        self.call_count += 1
    
    def _make_request(self, url: str, params: dict, max_retries: int = 3) -> requests.Response:
        """
        API 요청 실행 (재시도 로직 포함)
        
        Args:
            url (str): API 엔드포인트
            params (dict): 요청 파라미터
            max_retries (int): 최대 재시도 횟수
            
        Returns:
            requests.Response: API 응답
            
        Raises:
            requests.exceptions.RequestException: API 요청 실패
        """
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                self._rate_limit()
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response
                
            except requests.exceptions.Timeout:
                logger.warning(f"API 요청 타임아웃 (시도 {retry_count + 1}/{max_retries})")
                last_error = "Timeout"
                
            except requests.exceptions.ConnectionError:
                logger.warning(f"API 연결 오류 (시도 {retry_count + 1}/{max_retries})")
                last_error = "Connection Error"
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Too Many Requests
                    retry_after = int(e.response.headers.get('Retry-After', 60))
                    logger.warning(f"API 호출 제한 도달. {retry_after}초 후 재시도")
                    time.sleep(retry_after)
                    continue
                elif e.response.status_code == 404:  # Not Found
                    logger.error(f"API 요청 실패: {e}")
                    raise ValueError(f"잘못된 심볼 또는 인터벌: {e}")
                else:
                    logger.error(f"API HTTP 오류: {e}")
                    raise
                    
            except Exception as e:
                logger.error(f"API 요청 중 예외 발생: {e}")
                raise
                
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count  # 지수 백오프
                logger.info(f"{wait_time}초 후 재시도...")
                time.sleep(wait_time)
                
        raise requests.exceptions.RequestException(f"최대 재시도 횟수 초과. 마지막 오류: {last_error}")
    
    def _validate_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        데이터 검증 및 전처리
        
        Args:
            df (pd.DataFrame): 검증할 데이터
            symbol (str): 심볼
            
        Returns:
            pd.DataFrame: 검증 및 전처리된 데이터
        """
        try:
            if df is None or len(df) == 0:
                logger.warning(f"데이터가 비어있습니다: {symbol}")
                return pd.DataFrame()
                
            # 필수 컬럼 검증
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in df.columns]
                logger.error(f"필수 컬럼 누락: {missing_cols}")
                return pd.DataFrame()
                
            # 데이터 타입 변환
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # 결측치 처리
            if df.isnull().any().any():
                logger.warning(f"결측치 발견: {symbol}")
                df = df.fillna(method='ffill')  # 이전 값으로 채우기
                
            # 이상치 처리
            for col in ['open', 'high', 'low', 'close']:
                # 0 이하 값 처리
                df.loc[df[col] <= 0, col] = np.nan
                
                # Z-score 기반 이상치 처리
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df.loc[z_scores > 3, col] = np.nan
                
            # 결측치가 있는 행 제거
            df = df.dropna()
            
            # 가격 일관성 검증
            invalid_rows = (df['high'] < df['low']) | (df['open'] > df['high']) | (df['open'] < df['low']) | (df['close'] > df['high']) | (df['close'] < df['low'])
            if invalid_rows.any():
                logger.warning(f"가격 일관성 위반 행 발견: {symbol}")
                df = df[~invalid_rows]
                
            # 거래량 검증
            df.loc[df['volume'] < 0, 'volume'] = 0
            
            return df
            
        except Exception as e:
            logger.error(f"데이터 검증 중 오류 발생: {e}")
            return pd.DataFrame()
            
    def get_upbit_ohlcv(self, symbol: str, interval: str, count: int = 200) -> pd.DataFrame:
        """업비트 API에서 OHLCV 데이터를 가져옵니다."""
        try:
            # 캐시 키 생성
            cache_key = f"{symbol}_{interval}_{count}"
            
            # 캐시된 데이터 확인
            if cache_key in self._cache:
                cached_data = self._cache[cache_key]
                if not cached_data.empty:
                    return cached_data
            
            # interval에 따라 URL 결정
            if interval == "1d":
                url = "https://api.upbit.com/v1/candles/days"
            elif interval == "1w":
                url = "https://api.upbit.com/v1/candles/weeks"
            elif interval == "1M":
                url = "https://api.upbit.com/v1/candles/months"
            else:
                # 기본적으로 분봉 (예: 1h -> 60분)
                try:
                    minute = int(interval.replace("m", "").replace("h", ""))
                    if "h" in interval:
                        minute *= 60
                except Exception:
                    raise ValueError(f"지원하지 않는 interval: {interval}")
                url = f"https://api.upbit.com/v1/candles/minutes/{minute}"
            params = {"market": symbol, "count": count}
            
            # count 유효성 검사
            if count < 1:
                raise ValueError(f"count는 1 이상이어야 합니다: {count}")
            
            # API 요청
            response = self._make_request(url, params)
            data = response.json()
            
            if data is None or len(data) == 0:
                raise ValueError(f"데이터를 가져올 수 없습니다: {symbol}")
            
            # 데이터프레임 생성 및 전처리
            df = pd.DataFrame(data)
            
            # 타임스탬프 컬럼 처리
            if 'candle_date_time_kst' in df.columns:
                df['timestamp'] = pd.to_datetime(df['candle_date_time_kst'])
                df = df.drop('candle_date_time_kst', axis=1)
            elif 'candle_date_time_utc' in df.columns:
                df['timestamp'] = pd.to_datetime(df['candle_date_time_utc'])
                df = df.drop('candle_date_time_utc', axis=1)
            
            # 컬럼명 변경
            df = df.rename(columns={
                'opening_price': 'open',
                'high_price': 'high',
                'low_price': 'low',
                'trade_price': 'close',
                'candle_acc_trade_volume': 'volume'
            })
            
            # 필요한 컬럼만 선택
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # 캐시에 저장
            self._cache[cache_key] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Upbit API 요청 실패: {e}")
            raise
    
    def save_to_cache(self, df: pd.DataFrame, symbol: str, interval: str, cache_path: str):
        """데이터 캐시 저장"""
        try:
            os.makedirs(cache_path, exist_ok=True)
            file_path = os.path.join(cache_path, f"{symbol}_{interval}.csv")
            df.to_csv(file_path, encoding='utf-8')
        except Exception as e:
            self.logger.error(f"캐시 저장 중 오류 발생: {str(e)}")
    
    def load_from_cache(self, symbol: str, interval: str, cache_path: str, date: str = None) -> pd.DataFrame:
        """데이터 캐시 로드"""
        try:
            file_path = os.path.join(cache_path, f"{symbol}_{interval}.csv")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"캐시 파일이 존재하지 않습니다: {file_path}")
            
            df = pd.read_csv(file_path, index_col=0, parse_dates=True, encoding='utf-8')
            
            if date:
                df = df[df.index.date == pd.to_datetime(date).date()]
            
            return df
        except Exception as e:
            self.logger.error(f"캐시 로드 중 오류 발생: {str(e)}")
            raise
    
    def get_historical_data(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        """과거 데이터 조회"""
        try:
            # 기간 설정
            if period == "1d":
                count = 1
            elif period == "1w":
                count = 7
            elif period == "1mo":
                count = 31
            elif period == "3mo":
                count = 93
            elif period == "6mo":
                count = 186
            elif period == "1y":
                count = 365
            else:
                raise ValueError(f"지원하지 않는 기간입니다: {period}")
            
            # 데이터 조회
            df = self.get_upbit_ohlcv(symbol, interval, count)
            if df is None:
                raise ValueError("데이터 조회 실패: None 반환")
            
            return df
        except Exception as e:
            self.logger.error(f"과거 데이터 조회 중 오류 발생: {str(e)}")
            raise

if __name__ == "__main__":
    # 테스트 코드
    ds = DataSource()
    df = ds.get_historical_data("KRW-BTC", "1d")
    if df is not None:
        print(f"수집된 데이터 크기: {len(df)}")
        print(df.head())
    else:
        print("데이터 로드 실패") 