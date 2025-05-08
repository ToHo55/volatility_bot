import pytest
import pandas as pd
import numpy as np
from src.signals import SignalGenerator
from src.indicators import TechnicalIndicators

@pytest.fixture
def signal_generator():
    """SignalGenerator 인스턴스 생성"""
    return SignalGenerator()

@pytest.fixture
def sample_data():
    """샘플 OHLCV 데이터 생성"""
    # 상승 추세 데이터
    df = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [102, 103, 104, 105, 106],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }, index=pd.date_range(start='2024-01-01', periods=5, freq='H'))
    
    # 지표 추가
    indicators = TechnicalIndicators()
    return indicators.add_indicators(df)

def test_generate_signals(signal_generator, sample_data):
    """시그널 생성 테스트"""
    # 시그널 생성
    df = signal_generator.generate_signals(sample_data)
    
    # 검증
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_data)
    assert 'position' in df.columns
    assert 'stop_price' in df.columns
    assert 'stop_loss' in df.columns
    
    # 포지션 값 검증
    assert all(df['position'].isin([0, 1]))
    assert all(df['stop_loss'].isin([True, False]))

def test_calculate_stop_price(signal_generator, sample_data):
    """스탑 가격 계산 테스트"""
    # 롱 포지션 진입
    df = sample_data.copy()
    df['position'] = 1
    df['close'] = 100
    
    # 스탑 가격 계산
    stop_price = signal_generator._calculate_stop_price(df)
    
    # 검증
    assert isinstance(stop_price, pd.Series)
    assert len(stop_price) == len(df)
    assert all(stop_price > 0)
    assert all(stop_price < df['close'])

def test_check_stop_loss(signal_generator, sample_data):
    """스탑로스 체크 테스트"""
    # 롱 포지션 진입
    df = sample_data.copy()
    df['position'] = 1
    df['close'] = 100
    df['stop_price'] = 95
    
    # 스탑로스 체크
    stop_loss = signal_generator._check_stop_loss(df)
    
    # 검증
    assert isinstance(stop_loss, pd.Series)
    assert len(stop_loss) == len(df)
    assert all(stop_loss.isin([True, False]))

def test_entry_conditions(signal_generator, sample_data):
    """진입 조건 테스트"""
    # RSI 과매도 조건
    df = sample_data.copy()
    df['rsi'] = 20  # 과매도
    
    # EMA 상승 조건
    df['ema5'] = 100
    df['ema20'] = 90
    df['ema5_slope'] = True
    
    # 시그널 생성
    df = signal_generator.generate_signals(df)
    
    # 검증
    assert df['position'].iloc[-1] == 1  # 마지막 봉에서 롱 진입

def test_exit_conditions(signal_generator, sample_data):
    """청산 조건 테스트"""
    # RSI 과매수 조건
    df = sample_data.copy()
    df['rsi'] = 80  # 과매수
    
    # EMA 하락 조건
    df['ema5'] = 90
    df['ema20'] = 100
    df['ema5_slope'] = False
    
    # 시그널 생성
    df = signal_generator.generate_signals(df)
    
    # 검증
    assert df['position'].iloc[-1] == 0  # 마지막 봉에서 청산

def test_error_handling(signal_generator, sample_data):
    """에러 처리 테스트"""
    # 필수 컬럼 누락
    with pytest.raises(Exception):
        signal_generator.generate_signals(sample_data.drop('rsi', axis=1))
    
    # 잘못된 데이터 타입
    with pytest.raises(Exception):
        signal_generator.generate_signals(sample_data.astype(str)) 