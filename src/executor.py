import time
import uuid
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from loguru import logger
from rich.console import Console
from rich.table import Table
from .signals import SignalGenerator
from .datasource import DataSource

class Order:
    def __init__(self, order_id: str, symbol: str, side: str, 
                 quantity: float, price: float, order_type: str = "limit"):
        """
        주문 객체 초기화
        
        Args:
            order_id (str): 주문 ID
            symbol (str): 거래 심볼
            side (str): 매수/매도 구분
            quantity (float): 주문 수량
            price (float): 주문 가격
            order_type (str): 주문 유형 (limit/market)
        """
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.price = price
        self.order_type = order_type
        self.status = "pending"  # pending, filled, cancelled, rejected
        self.filled_quantity = 0.0
        self.filled_price = 0.0
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

class Position:
    def __init__(self, symbol: str, side: str, quantity: float, 
                 entry_price: float):
        """
        포지션 객체 초기화
        
        Args:
            symbol (str): 거래 심볼
            side (str): 매수/매도 구분
            quantity (float): 포지션 수량
            entry_price (float): 진입 가격
        """
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.entry_price = entry_price
        self.current_price = entry_price
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

class Executor:
    def __init__(self, api_key: str, secret_key: str, 
                 initial_capital: float = 1000000):
        """
        실행기 초기화
        
        Args:
            api_key (str): Upbit API 키
            secret_key (str): Upbit Secret 키
            initial_capital (float): 초기 자본금
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.signal_generator = SignalGenerator()
        self.data_source = DataSource()
        self.console = Console()
        
    def place_order(self, symbol: str, side: str, quantity: float, 
                   price: float, order_type: str = "limit") -> Order:
        """
        주문 실행
        
        Args:
            symbol (str): 거래 심볼
            side (str): 매수/매도 구분
            quantity (float): 주문 수량
            price (float): 주문 가격
            order_type (str): 주문 유형
            
        Returns:
            Order: 주문 객체
        """
        try:
            # 주문 ID 생성
            order_id = str(uuid.uuid4())
            
            # 주문 객체 생성
            order = Order(order_id, symbol, side, quantity, price, order_type)
            self.orders[order_id] = order
            
            # TODO: Upbit API 연동
            # 실제 거래소 API 호출 구현 필요
            
            logger.info(f"주문 실행: {order_id} - {symbol} {side} {quantity}@{price}")
            return order
            
        except Exception as e:
            logger.error(f"주문 실행 중 오류 발생: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        주문 취소
        
        Args:
            order_id (str): 주문 ID
            
        Returns:
            bool: 취소 성공 여부
        """
        try:
            if order_id not in self.orders:
                logger.error(f"존재하지 않는 주문: {order_id}")
                return False
            
            order = self.orders[order_id]
            if order.status != "pending":
                logger.error(f"취소할 수 없는 주문 상태: {order.status}")
                return False
            
            # TODO: Upbit API 연동
            # 실제 거래소 API 호출 구현 필요
            
            order.status = "cancelled"
            order.updated_at = datetime.now()
            logger.info(f"주문 취소: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"주문 취소 중 오류 발생: {e}")
            return False
    
    def update_position(self, symbol: str, current_price: float):
        """
        포지션 정보 업데이트
        
        Args:
            symbol (str): 거래 심볼
            current_price (float): 현재 가격
        """
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            position.current_price = current_price
            
            # PnL 계산
            if position.side == "buy":
                position.pnl = (current_price - position.entry_price) * position.quantity
                position.pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
            else:
                position.pnl = (position.entry_price - current_price) * position.quantity
                position.pnl_pct = (position.entry_price - current_price) / position.entry_price * 100
            
            position.updated_at = datetime.now()
            
        except Exception as e:
            logger.error(f"포지션 업데이트 중 오류 발생: {e}")
    
    def execute_strategy(self, symbol: str):
        """
        전략 실행
        
        Args:
            symbol (str): 거래 심볼
        """
        try:
            # 최신 데이터 조회
            df = self.data_source.get_historical_data(symbol, "1h", 100)
            if df is None or len(df) == 0:
                logger.error("데이터 조회 실패")
                return
            
            # 신호 생성
            df = self.signal_generator.generate_signals(df)
            latest = df.iloc[-1]
            
            # 현재 포지션 확인
            current_position = self.positions.get(symbol)
            
            # 포지션 진입/청산
            if latest['position'] == 1 and current_position is None:
                # 롱 포지션 진입
                quantity = self.current_capital * 0.95 / latest['close']  # 자본금의 95% 사용
                order = self.place_order(symbol, "buy", quantity, latest['close'])
                if order:
                    self.positions[symbol] = Position(symbol, "buy", quantity, latest['close'])
                    logger.info(f"롱 포지션 진입: {symbol} {quantity}@{latest['close']}")
            
            elif latest['position'] == 0 and current_position is not None:
                # 포지션 청산
                order = self.place_order(symbol, "sell", current_position.quantity, latest['close'])
                if order:
                    del self.positions[symbol]
                    logger.info(f"포지션 청산: {symbol} {current_position.quantity}@{latest['close']}")
            
            # 포지션 정보 업데이트
            if current_position is not None:
                self.update_position(symbol, latest['close'])
            
        except Exception as e:
            logger.error(f"전략 실행 중 오류 발생: {e}")
    
    def print_status(self):
        """
        현재 상태 출력
        """
        try:
            # 자본금 정보
            table = Table(title="계좌 상태")
            table.add_column("항목", style="cyan")
            table.add_column("값", style="magenta")
            
            table.add_row("초기 자본금", f"{self.initial_capital:,.0f} KRW")
            table.add_row("현재 자본금", f"{self.current_capital:,.0f} KRW")
            table.add_row("수익률", f"{(self.current_capital/self.initial_capital - 1)*100:.2f}%")
            
            self.console.print(table)
            
            # 포지션 정보
            if self.positions:
                table = Table(title="포지션")
                table.add_column("심볼", style="cyan")
                table.add_column("방향", style="magenta")
                table.add_column("수량", style="green")
                table.add_column("진입가", style="yellow")
                table.add_column("현재가", style="yellow")
                table.add_column("PnL", style="red")
                table.add_column("PnL%", style="red")
                
                for symbol, position in self.positions.items():
                    table.add_row(
                        symbol,
                        position.side,
                        f"{position.quantity:.8f}",
                        f"{position.entry_price:,.0f}",
                        f"{position.current_price:,.0f}",
                        f"{position.pnl:,.0f}",
                        f"{position.pnl_pct:.2f}%"
                    )
                
                self.console.print(table)
            
        except Exception as e:
            logger.error(f"상태 출력 중 오류 발생: {e}")

if __name__ == "__main__":
    # 테스트 코드
    executor = Executor("your_api_key", "your_secret_key")
    
    # 상태 출력
    executor.print_status()
    
    # 전략 실행
    executor.execute_strategy("BTC-KRW") 