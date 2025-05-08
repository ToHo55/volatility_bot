import os
import time
from datetime import datetime, timedelta
import pandas as pd
import pyupbit
from loguru import logger
from typing import Optional, Dict, List

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
        
    def _rate_limit(self):
        """API 호출 제한 준수"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < 60/self.upbit_rate_limit:
            time.sleep(60/self.upbit_rate_limit - time_since_last_call)
        
        self.last_call_time = time.time()
    
    def get_upbit_ohlcv(self, ticker: str, interval: str = "minute1", 
                        to: Optional[str] = None, count: int = 200) -> pd.DataFrame:
        """
        Upbit에서 OHLCV 데이터 조회
        
        Args:
            ticker (str): 코인 티커 (예: "KRW-BTC")
            interval (str): 시간 간격 (minute1, minute3, minute5, ...)
            to (str): 조회 종료 시간 (YYYY-MM-DD HH:mm:ss)
            count (int): 조회할 캔들 개수
            
        Returns:
            pd.DataFrame: OHLCV 데이터
        """
        self._rate_limit()
        
        try:
            df = pyupbit.get_ohlcv(ticker, interval=interval, to=to, count=count)
            if df is not None:
                df.index.name = 'timestamp'
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Upbit 데이터 조회 실패: {e}")
            return pd.DataFrame()
    
    def save_to_cache(self, df: pd.DataFrame, exchange: str, ticker: str, 
                     interval: str, date: str):
        """
        데이터를 CSV 캐시로 저장
        
        Args:
            df (pd.DataFrame): 저장할 데이터
            exchange (str): 거래소 이름
            ticker (str): 코인 티커
            interval (str): 시간 간격
            date (str): 날짜 (YYYY-MM-DD)
        """
        if df.empty:
            return
            
        filename = f"{exchange}_{ticker}_{interval}_{date}.csv"
        filepath = os.path.join(self.cache_dir, filename)
        df.to_csv(filepath)
        logger.info(f"데이터 캐시 저장 완료: {filepath}")
    
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
        filename = f"{exchange}_{ticker}_{interval}_{date}.csv"
        filepath = os.path.join(self.cache_dir, filename)
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
            logger.info(f"캐시에서 데이터 로드 완료: {filepath}")
            return df
        return pd.DataFrame()
    
    def get_historical_data(self, ticker: str, start_date: str, 
                           end_date: str, interval: str = "minute1") -> pd.DataFrame:
        """
        지정된 기간의 히스토리컬 데이터 조회
        
        Args:
            ticker (str): 코인 티커
            start_date (str): 시작 날짜 (YYYY-MM-DD)
            end_date (str): 종료 날짜 (YYYY-MM-DD)
            interval (str): 시간 간격
            
        Returns:
            pd.DataFrame: 히스토리컬 데이터
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_data = []
        current = start
        
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            
            # 캐시 확인
            cached_data = self.load_from_cache("upbit", ticker, interval, date_str)
            
            if not cached_data.empty:
                all_data.append(cached_data)
            else:
                # API로 데이터 조회
                df = self.get_upbit_ohlcv(ticker, interval=interval, 
                                        to=current.strftime("%Y-%m-%d %H:%M:%S"))
                if not df.empty:
                    self.save_to_cache(df, "upbit", ticker, interval, date_str)
                    all_data.append(df)
            
            current += timedelta(days=1)
        
        if all_data:
            return pd.concat(all_data)
        return pd.DataFrame()

if __name__ == "__main__":
    # 테스트 코드
    ds = DataSource()
    df = ds.get_historical_data("KRW-BTC", "2024-05-01", "2024-05-08")
    print(f"수집된 데이터 크기: {len(df)}")
    print(df.head()) 