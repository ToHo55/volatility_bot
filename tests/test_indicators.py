import pytest
import pandas as pd
import numpy as np
from src.indicators import TechnicalIndicators

@pytest.fixture
def indicators():
    """TechnicalIndicators 인스턴스 생성"""
    return TechnicalIndicators()

@pytest.fixture
def sample_data():
    """샘플 OHLCV 데이터 생성"""
    return pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [102, 103, 104, 105, 106],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }, index=pd.date_range(start='2024-01-01', periods=5, freq='H'))

def test_calc_rsi(indicators, sample_data):
    """RSI 계산 테스트"""
    # RSI 계산
    rsi = indicators.calc_rsi(sample_data['close'])
    
    # 검증
    assert isinstance(rsi, pd.Series)
    assert len(rsi) == len(sample_data)
    assert all(0 <= x <= 100 for x in rsi.dropna())
    assert rsi.name == 'rsi'

def test_calc_ema(indicators, sample_data):
    """EMA 계산 테스트"""
    # EMA 계산
    ema5 = indicators.calc_ema(sample_data['close'], 5)
    ema20 = indicators.calc_ema(sample_data['close'], 20)
    
    # 검증
    assert isinstance(ema5, pd.Series)
    assert isinstance(ema20, pd.Series)
    assert len(ema5) == len(sample_data)
    assert len(ema20) == len(sample_data)
    assert ema5.name == 'ema5'
    assert ema20.name == 'ema20'
    
    # EMA 값 검증
    assert all(ema5.notna())
    assert all(ema20.notna())
    assert all(ema5 > 0)
    assert all(ema20 > 0)

def test_ema_slope(indicators, sample_data):
    """EMA 기울기 계산 테스트"""
    # EMA 계산
    ema5 = indicators.calc_ema(sample_data['close'], 5)
    
    # 기울기 계산
    slope = indicators.ema_slope(ema5)
    
    # 검증
    assert isinstance(slope, pd.Series)
    assert len(slope) == len(sample_data)
    assert slope.name == 'ema5_slope'
    assert all(slope.isin([True, False]))

def test_calc_atr(indicators, sample_data):
    """ATR 계산 테스트"""
    # ATR 계산
    atr = indicators.calc_atr(sample_data)
    
    # 검증
    assert isinstance(atr, pd.Series)
    assert len(atr) == len(sample_data)
    assert atr.name == 'atr'
    assert all(atr > 0)
    assert all(atr.notna())

def test_add_indicators(indicators, sample_data):
    """모든 지표 추가 테스트"""
    # 지표 추가
    df = indicators.add_indicators(sample_data)
    
    # 검증
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_data)
    assert all(col in df.columns for col in ['rsi', 'ema5', 'ema20', 'ema5_slope', 'atr'])
    
    # 각 지표 값 검증
    assert all(0 <= x <= 100 for x in df['rsi'].dropna())
    assert all(df['ema5'].notna())
    assert all(df['ema20'].notna())
    assert all(df['ema5_slope'].isin([True, False]))
    assert all(df['atr'] > 0)

def test_error_handling(indicators, sample_data):
    """에러 처리 테스트"""
    # 잘못된 기간
    with pytest.raises(Exception):
        indicators.calc_rsi(sample_data['close'], period=0)
    
    # 잘못된 데이터
    with pytest.raises(Exception):
        indicators.calc_ema(pd.Series([np.nan, np.nan]), 5)
    
    # 잘못된 컬럼
    with pytest.raises(Exception):
        indicators.calc_atr(sample_data.drop('high', axis=1)) 