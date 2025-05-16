import asyncio
import json
import logging
import websockets
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
from .strategies import MLBoostSignal, MeanRevertSignal, BreakoutATRSignal

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UpbitBroker:
    """Upbit API 래퍼"""
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api.upbit.com/v1"
        
    async def get_balance(self) -> Dict[str, float]:
        """계좌 잔고 조회"""
        # TODO: Upbit API 연동
        return {"KRW": 1000000, "BTC": 0}
        
    async def place_order(self, market: str, side: str, volume: float = None, price: float = None):
        """주문 실행"""
        # TODO: Upbit API 연동
        logger.info(f"Order placed: {market} {side} {volume} {price}")
        
class PortfolioManager:
    """포트폴리오 관리"""
    
    def __init__(self, initial_weights: Dict[str, float]):
        self.weights = initial_weights
        self.positions: Dict[str, Dict[str, float]] = {}  # {strategy: {symbol: size}}
        self.equity = 0
        self.metrics = {
            'total_pnl': 0,
            'win_rate': 0,
            'avg_trade': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
        self.trade_history: List[Dict[str, Any]] = []
        
    def update_weights(self, new_weights: Dict[str, float]):
        """가중치 업데이트"""
        self.weights = new_weights
        
    def calculate_position_sizes(self, equity: float) -> Dict[str, float]:
        """포지션 크기 계산"""
        return {strategy: weight * equity for strategy, weight in self.weights.items()}
        
    def update_metrics(self, trade_result: Dict[str, Any]):
        """거래 결과로 지표 업데이트"""
        self.trade_history.append(trade_result)
        
        # 수익률 계산
        returns = [t['pnl_pct'] for t in self.trade_history]
        self.metrics['total_pnl'] = sum(returns)
        self.metrics['win_rate'] = len([r for r in returns if r > 0]) / len(returns)
        self.metrics['avg_trade'] = np.mean(returns)
        
        # 최대 낙폭
        cum_returns = np.cumsum(returns)
        rolling_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns - rolling_max
        self.metrics['max_drawdown'] = abs(min(drawdowns))
        
        # 샤프 비율
        if len(returns) > 1:
            self.metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
class Executor:
    """실시간 실행 엔진"""
    
    def __init__(self,
                 api_key: str,
                 secret_key: str,
                 strategies: Dict[str, Any],
                 initial_weights: Dict[str, float]):
        self.broker = UpbitBroker(api_key, secret_key)
        self.strategies = strategies
        self.portfolio = PortfolioManager(initial_weights)
        
    async def process_candle(self, candle: Dict[str, Any]):
        """캔들 데이터 처리"""
        try:
            # 데이터프레임 변환
            df = pd.DataFrame([candle])
            
            # 전략별 신호 생성
            signals = {}
            for name, strategy in self.strategies.items():
                if name == 'breakout_atr':
                    signal, size = strategy.generate_signal(df)
                else:
                    signal = strategy.generate_signal(df)
                    size = 1.0
                signals[name] = (signal.iloc[0], size)
                
            # 포지션 크기 계산
            balance = await self.broker.get_balance()
            self.portfolio.equity = balance['KRW']
            position_sizes = self.portfolio.calculate_position_sizes(self.portfolio.equity)
            
            # 주문 실행
            for name, (signal, _) in signals.items():
                if signal != 0:
                    size = position_sizes[name]
                    side = "bid" if signal > 0 else "ask"
                    await self.broker.place_order("KRW-BTC", side, volume=size)
                    
                    # 거래 기록
                    trade_result = {
                        'timestamp': datetime.now(),
                        'strategy': name,
                        'side': side,
                        'size': size,
                        'price': candle['trade_price'],
                        'pnl_pct': 0  # TODO: 실제 PnL 계산
                    }
                    self.portfolio.update_metrics(trade_result)
                    
        except Exception as e:
            logger.error(f"Error processing candle: {e}")
                
    async def run(self):
        """실행 루프"""
        uri = "wss://api.upbit.com/websocket/v1"
        
        while True:
            try:
                async with websockets.connect(uri) as websocket:
                    # 구독 메시지
                    subscribe_fmt = [
                        {"ticket": "UNIQUE_TICKET"},
                        {
                            "type": "trade",
                            "codes": ["KRW-BTC"],
                            "isOnlyRealtime": True
                        }
                    ]
                    await websocket.send(json.dumps(subscribe_fmt))
                    
                    while True:
                        data = await websocket.recv()
                        candle = json.loads(data)
                        await self.process_candle(candle)
                        
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(5)  # 재연결 전 대기
                
    def print_status(self):
        """현재 상태 출력"""
        logger.info("=== Portfolio Status ===")
        logger.info(f"Equity: {self.portfolio.equity:,.0f} KRW")
        logger.info("Metrics:")
        for metric, value in self.portfolio.metrics.items():
            logger.info(f"  {metric}: {value:.2f}")
        logger.info("Positions:")
        for strategy, positions in self.portfolio.positions.items():
            logger.info(f"  {strategy}: {positions}")
            
if __name__ == "__main__":
    # 전략 초기화
    strategies = {
        'ml_boost': MLBoostSignal(),
        'mean_revert': MeanRevertSignal(),
        'breakout_atr': BreakoutATRSignal()
    }
    
    # 초기 가중치 설정
    initial_weights = {
        'ml_boost': 0.6,
        'mean_revert': 0.25,
        'breakout_atr': 0.15
    }
    
    # 실행 엔진 초기화
    executor = Executor(
        api_key="YOUR_API_KEY",
        secret_key="YOUR_SECRET_KEY",
        strategies=strategies,
        initial_weights=initial_weights
    )
    
    # 실행
    asyncio.run(executor.run()) 