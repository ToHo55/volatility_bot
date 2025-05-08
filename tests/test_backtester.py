import pytest
import pandas as pd
import numpy as np
from src.backtester import Backtester, WalkForward
from src.signals import SignalGenerator

@pytest.fixture
def backtester():
    """Backtester 인스턴스 생성"""
    return Backtester(initial_capital=1000000)

@pytest.fixture
def walk_forward():
    """WalkForward 인스턴스 생성"""
    return WalkForward(train_days=30, test_days=7)

@pytest.fixture
def sample_data():
    """샘플 OHLCV 데이터 생성"""
    # 1년치 일봉 데이터
    df = pd.DataFrame({
        'open': np.random.uniform(100, 200, 365),
        'high': np.random.uniform(200, 300, 365),
        'low': np.random.uniform(50, 100, 365),
        'close': np.random.uniform(100, 200, 365),
        'volume': np.random.uniform(1000, 5000, 365)
    }, index=pd.date_range(start='2023-01-01', periods=365, freq='D'))
    
    # 지표와 시그널 추가
    signal_generator = SignalGenerator()
    return signal_generator.generate_signals(df)

def test_run_backtest(backtester, sample_data):
    """백테스트 실행 테스트"""
    # 백테스트 실행
    df, metrics = backtester.run(sample_data)
    
    # 검증
    assert isinstance(df, pd.DataFrame)
    assert isinstance(metrics, dict)
    assert len(df) == len(sample_data)
    assert all(col in df.columns for col in ['position', 'stop_price', 'stop_loss', 'pnl', 'equity'])
    
    # 메트릭 검증
    assert 'total_return' in metrics
    assert 'cagr' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'win_rate' in metrics
    assert 'avg_win' in metrics
    assert 'avg_loss' in metrics
    assert 'trade_count' in metrics

def test_simulate_trades(backtester, sample_data):
    """거래 시뮬레이션 테스트"""
    # 거래 시뮬레이션
    df = backtester._simulate_trades(sample_data)
    
    # 검증
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_data)
    assert all(col in df.columns for col in ['pnl', 'equity'])
    assert all(df['equity'] >= 0)
    assert all(df['equity'].notna())

def test_calculate_metrics(backtester, sample_data):
    """성과 지표 계산 테스트"""
    # 거래 시뮬레이션
    df = backtester._simulate_trades(sample_data)
    
    # 메트릭 계산
    metrics = backtester._calculate_metrics(df)
    
    # 검증
    assert isinstance(metrics, dict)
    assert all(key in metrics for key in [
        'total_return', 'cagr', 'sharpe_ratio', 'max_drawdown',
        'win_rate', 'avg_win', 'avg_loss', 'trade_count'
    ])
    assert all(isinstance(value, (int, float)) for value in metrics.values())
    assert metrics['max_drawdown'] <= 0
    assert 0 <= metrics['win_rate'] <= 1

def test_walk_forward(walk_forward, sample_data):
    """워크포워드 테스트"""
    # 워크포워드 테스트 실행
    metrics = walk_forward.run(sample_data)
    
    # 검증
    assert isinstance(metrics, dict)
    assert all(key in metrics for key in [
        'total_return', 'cagr', 'sharpe_ratio', 'max_drawdown',
        'win_rate', 'avg_win', 'avg_loss', 'trade_count'
    ])
    assert all(isinstance(value, (int, float)) for value in metrics.values())

def test_error_handling(backtester, sample_data):
    """에러 처리 테스트"""
    # 필수 컬럼 누락
    with pytest.raises(Exception):
        backtester.run(sample_data.drop('close', axis=1))
    
    # 잘못된 데이터 타입
    with pytest.raises(Exception):
        backtester.run(sample_data.astype(str))
    
    # 잘못된 초기 자본
    with pytest.raises(Exception):
        Backtester(initial_capital=-1000000) 