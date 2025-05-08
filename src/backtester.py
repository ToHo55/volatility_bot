import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from loguru import logger
from rich.console import Console
from rich.table import Table
from .signals import SignalGenerator

class Backtester:
    def __init__(self, initial_capital: float = 1000000,
                 commission: float = 0.04,
                 slippage: float = 0.05):
        """
        백테스터 초기화
        
        Args:
            initial_capital (float): 초기 자본금
            commission (float): 수수료 (%)
            slippage (float): 슬리피지 (%)
        """
        self.initial_capital = initial_capital
        self.commission = commission / 100
        self.slippage = slippage / 100
        self.signal_generator = SignalGenerator()
        self.console = Console()
        
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        백테스트 실행
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            
        Returns:
            Tuple[pd.DataFrame, Dict]: 백테스트 결과와 성과 지표
        """
        try:
            # 신호 생성
            df = self.signal_generator.generate_signals(df)
            
            # 포지션 시뮬레이션
            df = self._simulate_trades(df)
            
            # 성과 지표 계산
            metrics = self._calculate_metrics(df)
            
            return df, metrics
            
        except Exception as e:
            logger.error(f"백테스트 실행 중 오류 발생: {e}")
            return df, {}
    
    def _simulate_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        거래 시뮬레이션
        
        Args:
            df (pd.DataFrame): OHLCV + 신호 데이터
            
        Returns:
            pd.DataFrame: 거래 결과가 추가된 데이터프레임
        """
        try:
            # 초기 설정
            df['capital'] = self.initial_capital
            df['position_size'] = 0.0
            df['trade_pnl'] = 0.0
            df['commission_paid'] = 0.0
            df['slippage_cost'] = 0.0
            df['net_pnl'] = 0.0
            
            position = 0
            entry_price = 0.0
            
            for i in range(1, len(df)):
                # 이전 상태 복사
                df.iloc[i, df.columns.get_loc('capital')] = df.iloc[i-1]['capital']
                df.iloc[i, df.columns.get_loc('position_size')] = df.iloc[i-1]['position_size']
                
                # 포지션 변경 확인
                if df.iloc[i]['position'] != position:
                    # 청산
                    if position != 0:
                        exit_price = df.iloc[i]['close'] * (1 - self.slippage if position > 0 else 1 + self.slippage)
                        trade_pnl = (exit_price - entry_price) * position
                        commission = abs(position) * exit_price * self.commission
                        slippage_cost = abs(position) * exit_price * self.slippage
                        
                        df.iloc[i, df.columns.get_loc('trade_pnl')] = trade_pnl
                        df.iloc[i, df.columns.get_loc('commission_paid')] = commission
                        df.iloc[i, df.columns.get_loc('slippage_cost')] = slippage_cost
                        df.iloc[i, df.columns.get_loc('net_pnl')] = trade_pnl - commission - slippage_cost
                        df.iloc[i, df.columns.get_loc('capital')] += df.iloc[i]['net_pnl']
                        df.iloc[i, df.columns.get_loc('position_size')] = 0
                    
                    # 진입
                    if df.iloc[i]['position'] != 0:
                        position = df.iloc[i]['position']
                        entry_price = df.iloc[i]['close'] * (1 + self.slippage if position > 0 else 1 - self.slippage)
                        commission = abs(position) * entry_price * self.commission
                        slippage_cost = abs(position) * entry_price * self.slippage
                        
                        df.iloc[i, df.columns.get_loc('commission_paid')] = commission
                        df.iloc[i, df.columns.get_loc('slippage_cost')] = slippage_cost
                        df.iloc[i, df.columns.get_loc('net_pnl')] = -commission - slippage_cost
                        df.iloc[i, df.columns.get_loc('capital')] += df.iloc[i]['net_pnl']
                        df.iloc[i, df.columns.get_loc('position_size')] = position
            
            return df
            
        except Exception as e:
            logger.error(f"거래 시뮬레이션 중 오류 발생: {e}")
            return df
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """
        성과 지표 계산
        
        Args:
            df (pd.DataFrame): 백테스트 결과 데이터
            
        Returns:
            Dict: 성과 지표
        """
        try:
            # 수익률 계산
            total_return = (df['capital'].iloc[-1] - self.initial_capital) / self.initial_capital
            
            # 일별 수익률
            daily_returns = df['capital'].pct_change()
            
            # 연간화된 수익률 (CAGR)
            days = (df.index[-1] - df.index[0]).days
            cagr = (1 + total_return) ** (365/days) - 1
            
            # 샤프 비율
            risk_free_rate = 0.02  # 연 2% 가정
            excess_returns = daily_returns - risk_free_rate/252
            sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
            # 최대 낙폭 (MDD)
            cumulative_returns = (1 + daily_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            mdd = drawdowns.min()
            
            # 승률
            trades = df[df['trade_pnl'] != 0]
            win_rate = len(trades[trades['trade_pnl'] > 0]) / len(trades) if len(trades) > 0 else 0
            
            # 평균 수익/손실
            avg_win = trades[trades['trade_pnl'] > 0]['trade_pnl'].mean() if len(trades[trades['trade_pnl'] > 0]) > 0 else 0
            avg_loss = trades[trades['trade_pnl'] < 0]['trade_pnl'].mean() if len(trades[trades['trade_pnl'] < 0]) > 0 else 0
            
            return {
                'total_return': total_return,
                'cagr': cagr,
                'sharpe': sharpe,
                'mdd': mdd,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_trades': len(trades),
                'winning_trades': len(trades[trades['trade_pnl'] > 0]),
                'losing_trades': len(trades[trades['trade_pnl'] < 0])
            }
            
        except Exception as e:
            logger.error(f"성과 지표 계산 중 오류 발생: {e}")
            return {}
    
    def print_results(self, metrics: Dict):
        """
        백테스트 결과 출력
        
        Args:
            metrics (Dict): 성과 지표
        """
        table = Table(title="백테스트 결과")
        
        table.add_column("지표", style="cyan")
        table.add_column("값", style="magenta")
        
        table.add_row("총 수익률", f"{metrics['total_return']*100:.2f}%")
        table.add_row("CAGR", f"{metrics['cagr']*100:.2f}%")
        table.add_row("샤프 비율", f"{metrics['sharpe']:.2f}")
        table.add_row("최대 낙폭", f"{metrics['mdd']*100:.2f}%")
        table.add_row("승률", f"{metrics['win_rate']*100:.2f}%")
        table.add_row("평균 수익", f"{metrics['avg_win']:.2f}")
        table.add_row("평균 손실", f"{metrics['avg_loss']:.2f}")
        table.add_row("총 거래 횟수", str(metrics['total_trades']))
        table.add_row("승리 거래", str(metrics['winning_trades']))
        table.add_row("손실 거래", str(metrics['losing_trades']))
        
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
    
    def run(self, df: pd.DataFrame) -> Dict:
        """
        Walk-forward 테스트 실행
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            
        Returns:
            Dict: 전체 성과 지표
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
                    'avg_sharpe': np.mean([m['sharpe'] for m in all_metrics]),
                    'avg_win_rate': np.mean([m['win_rate'] for m in all_metrics]),
                    'total_trades': sum([m['total_trades'] for m in all_metrics]),
                    'winning_trades': sum([m['winning_trades'] for m in all_metrics]),
                    'losing_trades': sum([m['losing_trades'] for m in all_metrics])
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
    print(f"평균 수익률: {wf_metrics['avg_return']*100:.2f}%")
    print(f"평균 샤프 비율: {wf_metrics['avg_sharpe']:.2f}")
    print(f"평균 승률: {wf_metrics['avg_win_rate']*100:.2f}%")
    print(f"총 거래 횟수: {wf_metrics['total_trades']}")
    print(f"승리 거래: {wf_metrics['winning_trades']}")
    print(f"손실 거래: {wf_metrics['losing_trades']}") 