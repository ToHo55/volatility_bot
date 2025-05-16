import os
import time
from datetime import datetime, timedelta
import pandas as pd
import pyupbit
from loguru import logger
from typing import Optional, Dict, List
import requests
import numpy as np

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
            
    def get_upbit_ohlcv(self, symbol: str, interval: str, count: int) -> pd.DataFrame:
        """
        Upbit OHLCV 데이터 조회
        
        Args:
            symbol (str): 심볼 (예: BTC-KRW)
            interval (str): 시간 간격 (1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M)
            count (int): 조회할 캔들 개수
            
        Returns:
            pd.DataFrame: OHLCV 데이터
        """
        try:
            # API 엔드포인트
            url = f"https://api.upbit.com/v1/candles/{interval}"
            
            # 파라미터 설정
            params = {
                'market': symbol,
                'count': min(count, 200)  # 최대 200개로 제한
            }
            
            # API 요청
            response = self._make_request(url, params)
            
            # 데이터 변환
            data = response.json()
            if not data:
                logger.warning(f"데이터가 비어있습니다: {symbol}")
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            
            # 컬럼명 변경
            df = df.rename(columns={
                'opening_price': 'open',
                'high_price': 'high',
                'low_price': 'low',
                'trade_price': 'close',
                'candle_acc_trade_volume': 'volume'
            })
            
            # 인덱스 설정
            df['timestamp'] = pd.to_datetime(df['candle_date_time_kst'])
            df.set_index('timestamp', inplace=True)
            
            # 필요한 컬럼만 선택
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # 데이터 정렬
            df = df.sort_index()
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Upbit API 요청 실패: {e}")
            raise
        except ValueError as e:
            logger.error(f"데이터 변환 실패: {e}")
            raise
        except Exception as e:
            logger.error(f"Upbit 데이터 조회 실패: {e}")
            raise
    
    def save_to_cache(self, df: pd.DataFrame, symbol: str, interval: str, cache_path: str, date: str = None):
        """
        데이터를 캐시에 저장
        
        Args:
            df (pd.DataFrame): 저장할 데이터
            symbol (str): 심볼
            interval (str): 시간 간격
            cache_path (str): 캐시 파일 경로
            date (str, optional): 날짜 (기본값: 현재 날짜)
        """
        try:
            if date is None:
                date = datetime.now().strftime('%Y%m%d')
                
            # 캐시 파일 경로 생성
            cache_file = os.path.join(cache_path, f"{symbol}_{interval}_{date}.csv")
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            # 데이터 저장
            df.to_csv(cache_file)
            
        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")
            raise
    
    def load_from_cache(self, exchange: str, ticker: str, 
                       interval: str, date: str) -> pd.DataFrame:
        """
        CSV 캐시에서 데이터 로드
        
        Args:
            exchange (str): 거래소 이름
            ticker (str): 코인 티커
            interval (str): 시간 간격
            date (str): 날짜 (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: 캐시된 데이터
        """
        try:
            filename = f"{exchange}_{ticker}_{interval}_{date}.csv"
            filepath = os.path.join(self.cache_dir, filename)
            
            if not os.path.exists(filepath):
                logger.warning(f"캐시 파일이 존재하지 않습니다: {filepath}")
                return pd.DataFrame()
                
            # 데이터 로드
            df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
            
            # 데이터 검증
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"캐시 파일에 필수 컬럼이 누락되었습니다: {filepath}")
                return pd.DataFrame()
                
            # 데이터 정렬
            df = df.sort_index()
            
            logger.info(f"캐시에서 데이터 로드 완료: {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"캐시 로드 실패: {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, symbol: str, interval: str, 
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        과거 데이터 조회
        
        Args:
            symbol (str): 거래 심볼
            interval (str): 시간 간격 (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M)
            start_date (str, optional): 시작 날짜 (YYYY-MM-DD)
            end_date (str, optional): 종료 날짜 (YYYY-MM-DD)
            
        Returns:
            Optional[pd.DataFrame]: OHLCV 데이터
        """
        try:
            # 날짜 처리
            if start_date is None:
                if interval == "1d":
                    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                elif interval == "1h":
                    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                else:
                    start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                    
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
                
            # 날짜 파싱
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                # 상대적 기간 처리 (예: "1d", "1w", "1mo")
                if start_date.endswith("d"):
                    days = int(start_date[:-1])
                    start = datetime.now() - timedelta(days=days)
                elif start_date.endswith("w"):
                    weeks = int(start_date[:-1])
                    start = datetime.now() - timedelta(weeks=weeks)
                elif start_date.endswith("mo"):
                    months = int(start_date[:-2])
                    start = datetime.now() - timedelta(days=months*30)
                else:
                    raise ValueError(f"잘못된 날짜 형식: {start_date}")
                    
                end = datetime.now()
                
            # 데이터 조회
            days_diff = (end - start).days
            if days_diff > 200:  # 200일 이상인 경우 분할 조회
                dfs = []
                current_start = start
                while current_start < end:
                    current_end = min(current_start + timedelta(days=200), end)
                    df_chunk = self.get_upbit_ohlcv(symbol, interval, 200)
                    if df_chunk is not None and not df_chunk.empty:
                        dfs.append(df_chunk)
                    current_start = current_end
                df = pd.concat(dfs) if dfs else pd.DataFrame()
            else:
                df = self.get_upbit_ohlcv(symbol, interval, days_diff + 1)
            
            if df is None or len(df) == 0:
                logger.error(f"데이터 조회 실패: {symbol}")
                return None
                
            # 날짜 필터링
            df = df[(df.index >= start) & (df.index <= end)]
            
            # 중복 제거
            df = df[~df.index.duplicated(keep='first')]
            
            return df
            
        except Exception as e:
            logger.error(f"과거 데이터 조회 중 오류 발생: {e}")
            return None

if __name__ == "__main__":
    # 테스트 코드
    ds = DataSource()
    df = ds.get_historical_data("KRW-BTC", "1d")
    if df is not None:
        print(f"수집된 데이터 크기: {len(df)}")
        print(df.head())
    else:
        print("데이터 로드 실패") 