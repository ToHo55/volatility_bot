import pytest
import os
import shutil
from src.logger import Logger

@pytest.fixture
def logger():
    """Logger 인스턴스 생성"""
    return Logger(log_dir="test_logs")

@pytest.fixture(autouse=True)
def cleanup():
    """테스트 후 로그 디렉토리 정리"""
    yield
    if os.path.exists("test_logs"):
        shutil.rmtree("test_logs")

def test_logger_initialization(logger):
    """로거 초기화 테스트"""
    # 검증
    assert os.path.exists("test_logs")
    assert os.path.exists(os.path.join("test_logs", "trading.log"))
    assert os.path.exists(os.path.join("test_logs", "error.log"))

def test_print_startup(logger):
    """시작 메시지 출력 테스트"""
    # 설정 정보
    config = {
        "symbol": "BTC-KRW",
        "interval": "1h",
        "initial_capital": 1000000,
        "commission": 0.0005
    }
    
    # 시작 메시지 출력
    logger.print_startup(config)
    
    # 로그 파일 검증
    with open(os.path.join("test_logs", "trading.log"), "r") as f:
        log_content = f.read()
        assert "Starting trading bot" in log_content
        assert "BTC-KRW" in log_content
        assert "1h" in log_content
        assert "1000000" in log_content
        assert "0.0005" in log_content

def test_print_shutdown(logger):
    """종료 메시지 출력 테스트"""
    # 종료 메시지 출력
    logger.print_shutdown()
    
    # 로그 파일 검증
    with open(os.path.join("test_logs", "trading.log"), "r") as f:
        log_content = f.read()
        assert "Shutting down trading bot" in log_content

def test_print_error(logger):
    """에러 메시지 출력 테스트"""
    # 에러 발생
    error = ValueError("Test error message")
    
    # 에러 메시지 출력
    logger.print_error(error)
    
    # 로그 파일 검증
    with open(os.path.join("test_logs", "error.log"), "r") as f:
        log_content = f.read()
        assert "Test error message" in log_content
        assert "ValueError" in log_content

def test_print_trade(logger):
    """거래 정보 출력 테스트"""
    # 거래 정보
    trade_info = {
        "symbol": "BTC-KRW",
        "side": "buy",
        "quantity": 0.1,
        "price": 50000000,
        "timestamp": "2024-01-01 00:00:00"
    }
    
    # 거래 정보 출력
    logger.print_trade(trade_info)
    
    # 로그 파일 검증
    with open(os.path.join("test_logs", "trading.log"), "r") as f:
        log_content = f.read()
        assert "BTC-KRW" in log_content
        assert "buy" in log_content
        assert "0.1" in log_content
        assert "50000000" in log_content
        assert "2024-01-01 00:00:00" in log_content

def test_print_position(logger):
    """포지션 정보 출력 테스트"""
    # 포지션 정보
    position_info = {
        "symbol": "BTC-KRW",
        "side": "long",
        "quantity": 0.1,
        "entry_price": 50000000,
        "current_price": 55000000,
        "pnl": 500000,
        "pnl_pct": 0.1
    }
    
    # 포지션 정보 출력
    logger.print_position(position_info)
    
    # 로그 파일 검증
    with open(os.path.join("test_logs", "trading.log"), "r") as f:
        log_content = f.read()
        assert "BTC-KRW" in log_content
        assert "long" in log_content
        assert "0.1" in log_content
        assert "50000000" in log_content
        assert "55000000" in log_content
        assert "500000" in log_content
        assert "0.1" in log_content

def test_print_progress(logger):
    """진행 상황 출력 테스트"""
    # 진행 상황 메시지
    message = "Processing data..."
    
    # 진행 상황 출력
    logger.print_progress(message)
    
    # 로그 파일 검증
    with open(os.path.join("test_logs", "trading.log"), "r") as f:
        log_content = f.read()
        assert "Processing data..." in log_content

def test_log_rotation(logger):
    """로그 파일 로테이션 테스트"""
    # 대량의 로그 생성
    for i in range(1000):
        logger.print_progress(f"Test message {i}")
    
    # 로그 파일 검증
    log_files = os.listdir("test_logs")
    assert len(log_files) >= 2  # 최소 2개 이상의 로그 파일이 있어야 함 