import sys
import os
from datetime import datetime
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

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
        log_file = os.path.join(log_dir, f"trading_{datetime.now().strftime('%Y%m%d')}.log")
        
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
        
        # 파일 출력 설정
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="00:00",  # 매일 자정에 새 파일
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
        logger.info("트레이딩 봇 시작")
    
    def print_shutdown(self):
        """
        종료 메시지 출력
        """
        self.console.print(Panel("[bold red]트레이딩 봇 종료[/bold red]"))
        logger.info("트레이딩 봇 종료")
    
    def print_error(self, error: Exception):
        """
        에러 메시지 출력
        
        Args:
            error (Exception): 에러 객체
        """
        self.console.print(Panel(f"[bold red]에러 발생: {str(error)}[/bold red]"))
        logger.exception(error)
    
    def print_trade(self, trade_info: dict):
        """
        거래 정보 출력
        
        Args:
            trade_info (dict): 거래 정보
        """
        table = Table(title="거래 실행")
        table.add_column("항목", style="cyan")
        table.add_column("값", style="magenta")
        
        for key, value in trade_info.items():
            table.add_row(str(key), str(value))
        
        self.console.print(table)
        logger.info(f"거래 실행: {trade_info}")
    
    def print_position(self, position_info: dict):
        """
        포지션 정보 출력
        
        Args:
            position_info (dict): 포지션 정보
        """
        table = Table(title="포지션 상태")
        table.add_column("항목", style="cyan")
        table.add_column("값", style="magenta")
        
        for key, value in position_info.items():
            table.add_row(str(key), str(value))
        
        self.console.print(table)
        logger.info(f"포지션 상태: {position_info}")
    
    def print_progress(self, message: str):
        """
        진행 상태 출력
        
        Args:
            message (str): 진행 메시지
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            progress.add_task(description=message, total=None)
            logger.info(message)

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