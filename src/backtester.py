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
        
    def run(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        백테스트 실행
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, float]]: 백테스트 결과와 성과 지표
        """
        try:
            # 지표 추가
            df = self.indicators.add_indicators(data)
            
            # 신호 생성
            df = self.signal_generator.generate_signals(df)
            
            # 거래 시뮬레이션
            df = self._simulate_trades(df)
            
            # 성과 지표 계산
            metrics = self._calculate_metrics(df)
            
            return df, metrics
            
        except Exception as e:
            logger.error(f"백테스트 실행 중 오류 발생: {e}")
            raise
    
    def _simulate_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        거래 시뮬레이션
        
        Args:
            df (pd.DataFrame): 신호가 포함된 데이터
            
        Returns:
            pd.DataFrame: 거래 결과가 추가된 데이터
        """
        try:
            # 초기 설정
            df['position'] = 0  # 포지션 (1: 롱, -1: 숏, 0: 중립)
            df['stop_price'] = np.nan  # 손절가
            df['stop_loss'] = np.nan  # 손절 여부
            df['pnl'] = 0.0  # 수익/손실
            df['equity'] = self.initial_capital  # 자본금
            
            position = 0
            entry_price = 0.0
            stop_price = 0.0
            
            for i in range(1, len(df)):
                # 이전 포지션 유지
                df.loc[df.index[i], 'position'] = position
                df.loc[df.index[i], 'equity'] = df.loc[df.index[i-1], 'equity']
                
                # 신호에 따른 포지션 변경
                if df.loc[df.index[i], 'signal'] == 1 and position <= 0:
                    # 롱 진입
                    position = 1
                    entry_price = df.loc[df.index[i], 'close']
                    stop_price = entry_price * 0.95  # 5% 손절
                    df.loc[df.index[i], 'position'] = position
                    df.loc[df.index[i], 'stop_price'] = stop_price
                    
                elif df.loc[df.index[i], 'signal'] == -1 and position >= 0:
                    # 숏 진입
                    position = -1
                    entry_price = df.loc[df.index[i], 'close']
                    stop_price = entry_price * 1.05  # 5% 손절
                    df.loc[df.index[i], 'position'] = position
                    df.loc[df.index[i], 'stop_price'] = stop_price
                    
                # 손절 체크
                if position != 0:
                    current_price = df.loc[df.index[i], 'close']
                    if (position == 1 and current_price <= stop_price) or \
                       (position == -1 and current_price >= stop_price):
                        # 손절
                        df.loc[df.index[i], 'stop_loss'] = True
                        position = 0
                        entry_price = 0.0
                        stop_price = 0.0
                        df.loc[df.index[i], 'position'] = position
                        df.loc[df.index[i], 'stop_price'] = np.nan
                        
                # 수익/손실 계산
                if position != 0:
                    pnl = (df.loc[df.index[i], 'close'] - entry_price) * position
                    commission = abs(pnl) * self.commission
                    slippage = abs(pnl) * self.slippage
                    net_pnl = pnl - commission - slippage
                    
                    df.loc[df.index[i], 'pnl'] = net_pnl
                    df.loc[df.index[i], 'equity'] = df.loc[df.index[i-1], 'equity'] + net_pnl
            
            return df
            
        except Exception as e:
            logger.error(f"거래 시뮬레이션 중 오류 발생: {e}")
            raise
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        성과 지표 계산
        
        Args:
            df (pd.DataFrame): 거래 결과가 포함된 데이터
            
        Returns:
            Dict[str, float]: 성과 지표
        """
        try:
            # 수익률 계산
            total_return = (df['equity'].iloc[-1] / self.initial_capital - 1) * 100
            
            # 연간 수익률 (CAGR)
            days = (df.index[-1] - df.index[0]).days
            cagr = ((1 + total_return/100) ** (365/days) - 1) * 100
            
            # 일간 수익률
            daily_returns = df['equity'].pct_change().dropna()
            
            # 샤프 비율
            risk_free_rate = 0.02  # 연 2% 무위험 수익률 가정
            excess_returns = daily_returns - risk_free_rate/252
            sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
            # 최대 낙폭
            cumulative_returns = (1 + daily_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min() * 100
            
            # 승률
            trades = df[df['position'] != df['position'].shift(1)]
            winning_trades = trades[trades['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades) * 100 if len(trades) > 0 else 0
            
            # 평균 수익/손실
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            losing_trades = trades[trades['pnl'] < 0]
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            
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
            logger.error(f"성과 지표 계산 중 오류 발생: {e}")
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
    df, metrics = bt.run(data)
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