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
        
        # 로그 파일 경로
        trading_log = os.path.join(log_dir, "trading.log")
        error_log = os.path.join(log_dir, "error.log")
        
        # 기존 핸들러 제거
        logger.remove()
        
        # 콘솔 출력 설정
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO",
            backtrace=True,
            diagnose=True
        )
        
        # 거래 로그 파일 설정
        logger.add(
            trading_log,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="INFO",
            rotation="1 MB",  # 1MB마다 새 파일
            retention="30 days",  # 30일 보관
            compression="zip",  # 압축 저장
            backtrace=True,
            diagnose=True
        )
        
        # 에러 로그 파일 설정
        logger.add(
            error_log,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR",
            rotation="1 MB",  # 1MB마다 새 파일
            retention="30 days",  # 30일 보관
            compression="zip",  # 압축 저장
            backtrace=True,
            diagnose=True
        )
        
        # rich 핸들러 설정
        logger.add(
            RichHandler(rich_tracebacks=True, markup=True),
            format="{message}",
            level="INFO"
        )
        
        # 로거 설정
        self.logger = logging.getLogger("trading_bot")
        self.logger.setLevel(logging.INFO)
        
        # 파일 핸들러 설정
        file_handler = logging.FileHandler(trading_log)
        file_handler.setLevel(logging.INFO)
        
        # 포맷터 설정
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 핸들러 추가
        self.logger.addHandler(file_handler)
    
    def print_startup(self, config: dict):
        """
        시작 메시지 출력
        
        Args:
            config (dict): 설정 정보
        """
        table = Table(title="트레이딩 봇 시작")
        table.add_column("항목", style="cyan")
        table.add_column("값", style="magenta")
        
        for key, value in config.items():
            table.add_row(str(key), str(value))
        
        self.console.print(Panel(table, title="[bold green]시작[/bold green]"))
        logger.info("Starting trading bot")
        for key, value in config.items():
            logger.info(f"{key}: {value}")
    
    def print_shutdown(self):
        """
        종료 메시지 출력
        """
        self.console.print(Panel("[bold red]트레이딩 봇 종료[/bold red]"))
        logger.info("Shutting down trading bot")
    
    def print_error(self, error: Exception):
        """에러 메시지 출력"""
        error_msg = f"에러 발생: {str(error)}"
        print(error_msg)
        
        # 에러 로그 파일에 기록
        with open(os.path.join(self.log_dir, "error.log"), "a", encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
        
        # 로거에도 기록
        self.logger.error(str(error))
    
    def print_trade(self, trade_info: dict):
        """거래 정보 출력"""
        trade_msg = (
            f"거래 실행: {trade_info['symbol']} | "
            f"방향: {trade_info['side']} | "
            f"수량: {trade_info['quantity']} | "
            f"가격: {trade_info['price']}"
        )
        print(trade_msg)
        
        # 거래 로그 파일에 기록
        with open(os.path.join(self.log_dir, "trading.log"), "a", encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {trade_msg}\n")
        
        # 로거에도 기록
        self.logger.info(trade_msg)
    
    def print_position(self, position_info: dict):
        """포지션 정보 출력"""
        position_msg = (
            f"포지션 상태: {position_info}"
        )
        print(position_msg)
        
        # 거래 로그 파일에 기록
        with open(os.path.join(self.log_dir, "trading.log"), "a", encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {position_msg}\n")
        
        # 로거에도 기록
        self.logger.info(position_msg)
    
    def print_progress(self, message: str):
        """진행 상황 출력"""
        print(message)
        
        # 거래 로그 파일에 기록
        with open(os.path.join(self.log_dir, "trading.log"), "a", encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        
        # 로거에도 기록
        self.logger.info(message)

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