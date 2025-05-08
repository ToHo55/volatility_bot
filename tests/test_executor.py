import pytest
from src.executor import Executor
from src.datasource import DataSource
from src.signals import SignalGenerator
from src.logger import Logger

class TestExecutor:
    @pytest.fixture
    def setup_executor(self):
        """테스트를 위한 Executor 인스턴스 설정"""
        logger = Logger()
        datasource = DataSource()
        signal_generator = SignalGenerator(datasource)
        return Executor(datasource, signal_generator, logger)

    def test_executor_initialization(self, setup_executor):
        """실행기 초기화 테스트"""
        executor = setup_executor
        assert executor is not None
        assert executor.datasource is not None
        assert executor.signal_generator is not None
        assert executor.logger is not None

    def test_execute_trade(self, setup_executor):
        """거래 실행 테스트"""
        executor = setup_executor
        # 테스트용 거래 데이터
        test_trade = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 0.1,
            'price': 50000
        }
        
        result = executor.execute_trade(test_trade)
        assert result is not None
        assert 'status' in result
        assert 'trade_id' in result

    def test_validate_trade(self, setup_executor):
        """거래 유효성 검사 테스트"""
        executor = setup_executor
        # 유효한 거래 데이터
        valid_trade = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 0.1,
            'price': 50000
        }
        
        # 유효하지 않은 거래 데이터
        invalid_trade = {
            'symbol': 'BTC/USDT',
            'side': 'invalid',
            'amount': -0.1,
            'price': -50000
        }
        
        assert executor.validate_trade(valid_trade) is True
        assert executor.validate_trade(invalid_trade) is False

    def test_handle_error(self, setup_executor):
        """에러 처리 테스트"""
        executor = setup_executor
        test_error = Exception("Test error")
        
        result = executor.handle_error(test_error)
        assert result is not None
        assert 'error' in result
        assert 'timestamp' in result 