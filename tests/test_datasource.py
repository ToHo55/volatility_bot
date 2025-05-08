import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.datasource import DataSource

@pytest.fixture
def datasource():
    """DataSource 인스턴스 생성"""
    return DataSource()

def test_get_upbit_ohlcv(datasource):
    """Upbit OHLCV 데이터 조회 테스트"""
    # 테스트 데이터
    symbol = "BTC-KRW"
    interval = "1h"
    count = 10
    
    # 데이터 조회
    df = datasource.get_upbit_ohlcv(symbol, interval, count)
    
    # 검증
    assert isinstance(df, pd.DataFrame)
    assert len(df) == count
    assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    assert df.index.is_monotonic_increasing

def test_save_and_load_cache(datasource, tmp_path):
    """캐시 저장 및 로드 테스트"""
    # 테스트 데이터
    symbol = "BTC-KRW"
    interval = "1h"
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [103, 104, 105],
        'low': [98, 99, 100],
        'close': [102, 103, 104],
        'volume': [1000, 1100, 1200]
    }, index=pd.date_range(start='2024-01-01', periods=3, freq='H'))
    
    # 캐시 저장
    cache_path = tmp_path / "cache"
    datasource.save_to_cache(df, symbol, interval, cache_path)
    
    # 캐시 로드
    loaded_df = datasource.load_from_cache(symbol, interval, cache_path)
    
    # 검증
    assert isinstance(loaded_df, pd.DataFrame)
    assert len(loaded_df) == len(df)
    assert all(col in loaded_df.columns for col in df.columns)
    pd.testing.assert_frame_equal(df, loaded_df)

def test_get_historical_data(datasource):
    """히스토리컬 데이터 조회 테스트"""
    # 테스트 데이터
    symbol = "BTC-KRW"
    interval = "1h"
    days = 7
    
    # 데이터 조회
    df = datasource.get_historical_data(symbol, interval, days)
    
    # 검증
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    assert df.index.is_monotonic_increasing
    
    # 날짜 범위 검증
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    assert df.index[0] >= start_date
    assert df.index[-1] <= end_date

def test_error_handling(datasource):
    """에러 처리 테스트"""
    # 잘못된 심볼
    with pytest.raises(Exception):
        datasource.get_upbit_ohlcv("INVALID-SYMBOL", "1h", 10)
    
    # 잘못된 인터벌
    with pytest.raises(Exception):
        datasource.get_upbit_ohlcv("BTC-KRW", "invalid", 10)
    
    # 잘못된 개수
    with pytest.raises(Exception):
        datasource.get_upbit_ohlcv("BTC-KRW", "1h", -1)

def test_datasource_initialization():
    datasource = DataSource()
    assert datasource is not None

def test_get_historical_data():
    datasource = DataSource()
    df = datasource.get_historical_data("BTC-USD", "1d", "1mo")
    assert df is not None
    assert not df.empty 