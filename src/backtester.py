import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from loguru import logger
from rich.console import Console
from rich.table import Table
from .indicators import TechnicalIndicators
from .signals import SignalGenerator

class Backtester:
    def __init__(self, initial_capital: float = 1000000, 
                 commission: float = 0.0005, slippage: float = 0.0001):
        """
        백테스터 초기화
        
        Args:
            initial_capital (float): 초기 자본금
            commission (float): 수수료율
            slippage (float): 슬리피지율
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.indicators = TechnicalIndicators()
        self.signal_generator = SignalGenerator()
        self.console = Console()
        
    def run(self, data: pd.DataFrame) -> dict:
        """백테스트 실행"""
        try:
            # 필수 컬럼 확인
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'position', 'stop_price', 'stop_hit']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"필수 컬럼이 누락되었습니다: {required_columns}")
            
            # 데이터 타입 확인
            if not all(data[col].dtype in [np.float64, np.int64] for col in ['open', 'high', 'low', 'close', 'volume']):
                raise ValueError("가격 데이터는 숫자형이어야 합니다")
            
            # 초기 자본 설정
            if self.initial_capital <= 0:
                raise ValueError("초기 자본은 0보다 커야 합니다")
            
            # 백테스트 실행
            self._run_backtest(data)
            
            # 성과 지표 계산
            metrics = self._calculate_metrics()
            
            return metrics
        except Exception as e:
            logger.error(f"백테스트 실행 중 오류 발생: {str(e)}")
            raise

    def _run_backtest(self, df: pd.DataFrame):
        """백테스트 실행 로직"""
        try:
            # 포트폴리오 초기화
            self.portfolio = {
                'cash': self.initial_capital,
                'position': 0,
                'entry_price': 0,
                'trades': []
            }
            
            # 일별 수익률 계산
            self.daily_returns = pd.Series(index=df.index, dtype=float)
            
            for i in range(len(df)):
                current_price = df['close'].iloc[i]
                position = df['position'].iloc[i]
                stop_hit = df['stop_hit'].iloc[i]
                
                # 포지션 진입
                if position == 1 and self.portfolio['position'] == 0:
                    self._enter_position(current_price)
                
                # 포지션 청산
                elif (position == 0 or stop_hit) and self.portfolio['position'] != 0:
                    self._exit_position(current_price)
                
                # 일별 수익률 계산
                if self.portfolio['position'] != 0:
                    self.daily_returns.iloc[i] = (current_price - self.portfolio['entry_price']) / self.portfolio['entry_price']
                else:
                    self.daily_returns.iloc[i] = 0
        except Exception as e:
            logger.error(f"백테스트 실행 중 오류 발생: {str(e)}")
            raise

    def _calculate_metrics(self) -> dict:
        """성과 지표 계산"""
        try:
            # 총 수익률
            total_return = (self.portfolio['cash'] - self.initial_capital) / self.initial_capital
            
            # 연간 수익률 (CAGR)
            days = (self.daily_returns.index[-1] - self.daily_returns.index[0]).days
            cagr = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
            
            # 샤프 비율
            excess_returns = self.daily_returns - 0.02/252  # 무위험 수익률 가정
            sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
            
            # 최대 낙폭
            cumulative_returns = (1 + self.daily_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            # 승률
            trades = self.portfolio['trades']
            if trades:
                winning_trades = [t for t in trades if t['pnl'] > 0]
                win_rate = len(winning_trades) / len(trades)
                avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
                avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] <= 0]) if any(t['pnl'] <= 0 for t in trades) else 0
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
            
            return {
                'total_return': total_return,
                'cagr': cagr,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'trade_count': len(trades)
            }
        except Exception as e:
            logger.error(f"성과 지표 계산 중 오류 발생: {str(e)}")
            raise

    def print_results(self, metrics: Dict[str, float]):
        """
        백테스트 결과 출력
        
        Args:
            metrics (Dict[str, float]): 성과 지표
        """
        table = Table(title="백테스트 결과")
        
        table.add_column("지표", style="cyan")
        table.add_column("값", style="magenta")
        
        table.add_row("총 수익률", f"{metrics['total_return']:.2f}%")
        table.add_row("CAGR", f"{metrics['cagr']:.2f}%")
        table.add_row("샤프 비율", f"{metrics['sharpe_ratio']:.2f}")
        table.add_row("최대 낙폭", f"{metrics['max_drawdown']:.2f}%")
        table.add_row("승률", f"{metrics['win_rate']:.2f}%")
        table.add_row("평균 수익", f"{metrics['avg_win']:.2f}")
        table.add_row("평균 손실", f"{metrics['avg_loss']:.2f}")
        table.add_row("총 거래 횟수", str(metrics['trade_count']))
        
        self.console.print(table)

class WalkForward:
    def __init__(self, train_days: int = 60, test_days: int = 14):
        """
        Walk-forward 테스트 초기화
        
        Args:
            train_days (int): 학습 기간 (일)
            test_days (int): 테스트 기간 (일)
        """
        self.train_days = train_days
        self.test_days = test_days
        self.backtester = Backtester()
    
    def run(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Walk-forward 테스트 실행
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            
        Returns:
            Dict[str, float]: 전체 성과 지표
        """
        try:
            all_metrics = []
            current_date = df.index[0]
            end_date = df.index[-1]
            
            while current_date + timedelta(days=self.train_days + self.test_days) <= end_date:
                # 학습 기간
                train_start = current_date
                train_end = train_start + timedelta(days=self.train_days)
                
                # 테스트 기간
                test_start = train_end
                test_end = test_start + timedelta(days=self.test_days)
                
                # 테스트 실행
                test_data = df[test_start:test_end]
                _, metrics = self.backtester.run(test_data)
                
                if metrics:
                    all_metrics.append(metrics)
                
                # 다음 기간으로 이동
                current_date += timedelta(days=self.test_days)
            
            # 전체 성과 지표 계산
            if all_metrics:
                return {
                    'avg_return': np.mean([m['total_return'] for m in all_metrics]),
                    'avg_sharpe': np.mean([m['sharpe_ratio'] for m in all_metrics]),
                    'avg_win_rate': np.mean([m['win_rate'] for m in all_metrics]),
                    'total_trades': sum([m['trade_count'] for m in all_metrics]),
                    'winning_trades': sum([m['trade_count'] for m in all_metrics if m['win_rate'] > 50]),
                    'losing_trades': sum([m['trade_count'] for m in all_metrics if m['win_rate'] <= 50])
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Walk-forward 테스트 중 오류 발생: {e}")
            return {}

if __name__ == "__main__":
    # 테스트 코드
    import yfinance as yf
    
    # 테스트 데이터 다운로드
    data = yf.download("BTC-USD", start="2024-01-01", end="2024-05-08", interval="1h")
    
    # 백테스트 실행
    bt = Backtester()
    metrics = bt.run(data)
    bt.print_results(metrics)
    
    # Walk-forward 테스트 실행
    wf = WalkForward()
    wf_metrics = wf.run(data)
    
    print("\nWalk-forward 테스트 결과:")
    print(f"평균 수익률: {wf_metrics['avg_return']:.2f}%")
    print(f"평균 샤프 비율: {wf_metrics['avg_sharpe']:.2f}")
    print(f"평균 승률: {wf_metrics['avg_win_rate']:.2f}%")
    print(f"총 거래 횟수: {wf_metrics['total_trades']}")
    print(f"승리 거래: {wf_metrics['winning_trades']}")
    print(f"손실 거래: {wf_metrics['losing_trades']}") 