import sys
import os
from datetime import datetime
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import logging
from logging.handlers import RotatingFileHandler

class Logger:
    def __init__(self, log_dir: str = "logs"):
        """
        로거 초기화
        
        Args:
            log_dir (str): 로그 파일 저장 디렉토리
        """
        self.log_dir = log_dir
        self.console = Console()
        
        # 로그 디렉토리 생성
        os.makedirs(log_dir, exist_ok=True)
        
        # 로그 파일 핸들러 설정
        self.trading_handler = RotatingFileHandler(
            os.path.join(log_dir, "trading.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        self.error_handler = RotatingFileHandler(
            os.path.join(log_dir, "error.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        # 포맷터 설정
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.trading_handler.setFormatter(formatter)
        self.error_handler.setFormatter(formatter)
        
        # 로거 설정
        self.logger = logging.getLogger("trading_bot")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.trading_handler)
        self.logger.addHandler(self.error_handler)
    
    def print_startup(self, config: dict):
        """
        시작 메시지 출력
        
        Args:
            config (dict): 설정 정보
        """
        try:
            # ASCII 아트 출력
            print("┌─────────────────────────────────── 시작 ────────────────────────────────────┐")
            print("│       트레이딩 봇 시작                                                      │")
            print("│ ┌─────────────────┬─────────┐                                               │")
            print("│ │ 항목            │ 값      │                                               │")
            print("│ ├─────────────────┼─────────┤                                               │")
            
            # 설정 정보 출력
            for key, value in config.items():
                print(f"│ │ {key:<15} │ {value:<7} │                                               │")
            
            print("│ └─────────────────┴─────────┘                                               │")
            print("└─────────────────────────────────────────────────────────────────────────────┘")
            
            # 로그 기록
            self.logger.info("Starting trading bot")
            for key, value in config.items():
                self.logger.info(f"{key}: {value}")
        except Exception as e:
            self.logger.error(f"시작 메시지 출력 중 오류 발생: {str(e)}")
    
    def print_shutdown(self):
        """
        종료 메시지 출력
        """
        try:
            print("┌─────────────────────────────────────────────────────────────────────────────┐")
            print("│ 트레이딩 봇 종료                                                            │")
            print("└─────────────────────────────────────────────────────────────────────────────┘")
            self.logger.info("Shutting down trading bot")
        except Exception as e:
            self.logger.error(f"종료 메시지 출력 중 오류 발생: {str(e)}")
    
    def print_error(self, error: Exception):
        """에러 메시지 출력"""
        try:
            print(f"에러 발생: {str(error)}")
            self.logger.error(str(error))
        except Exception as e:
            self.logger.error(f"에러 메시지 출력 중 오류 발생: {str(e)}")
    
    def print_trade(self, trade_info: dict):
        """거래 정보 출력"""
        try:
            print(f"거래 실행: {trade_info['symbol']} | 방향: {trade_info['side']} | 수량: {trade_info['quantity']} | 가격: {trade_info['price']}")
            self.logger.info(f"거래 실행: {trade_info['symbol']} | 방향: {trade_info['side']} | 수량: {trade_info['quantity']} | 가격: {trade_info['price']}")
        except Exception as e:
            self.logger.error(f"거래 정보 출력 중 오류 발생: {str(e)}")
    
    def print_position(self, position_info: dict):
        """포지션 정보 출력"""
        try:
            print(f"포지션 상태: {position_info}")
            self.logger.info(f"포지션 상태: {position_info}")
        except Exception as e:
            self.logger.error(f"포지션 정보 출력 중 오류 발생: {str(e)}")
    
    def print_progress(self, message: str):
        """진행 상황 출력"""
        try:
            print(message)
            self.logger.info(message)
        except Exception as e:
            self.logger.error(f"진행 상황 출력 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    # 테스트 코드
    logger = Logger()
    
    # 시작 메시지
    config = {
        "API_KEY": "your_api_key",
        "SECRET_KEY": "your_secret_key",
        "INITIAL_CAPITAL": 1000000,
        "TRADING_PAIRS": ["BTC-KRW", "ETH-KRW"]
    }
    logger.print_startup(config)
    
    # 거래 정보
    trade_info = {
        "symbol": "BTC-KRW",
        "side": "buy",
        "quantity": 0.1,
        "price": 50000000
    }
    logger.print_trade(trade_info)
    
    # 포지션 정보
    position_info = {
        "symbol": "BTC-KRW",
        "side": "buy",
        "quantity": 0.1,
        "entry_price": 50000000,
        "current_price": 51000000,
        "pnl": 100000,
        "pnl_pct": 2.0
    }
    logger.print_position(position_info)
    
    # 진행 상태
    logger.print_progress("데이터 수집 중...")
    
    # 에러 메시지
    try:
        1/0
    except Exception as e:
        logger.print_error(e)
    
    # 종료 메시지
    logger.print_shutdown() 