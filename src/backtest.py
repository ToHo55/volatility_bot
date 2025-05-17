import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import sys
import os

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

print(f"프로젝트 루트 디렉토리: {project_root}")
print(f"파이썬 경로: {sys.path}")

from strategies.mean_revert.signal import MeanRevertSignal
from strategies.ml_boost.ml_signal import MLBoostSignal

console = Console()

class BacktestResult:
    def __init__(self, strategy_name: str, initial_balance: float = 1000000, trade_ratio: float = 0.1):
        self.strategy_name = strategy_name
        self.trades = []
        self.initial_balance = initial_balance
        self.trade_ratio = trade_ratio  # 거래 자본금 비율
        self.trade_amount = initial_balance * trade_ratio  # 각 거래마다 사용할 자본금
        self.total_pnl = 0
        self.win_trades = 0
        self.loss_trades = 0
        self.max_drawdown = 0
        self.peak_balance = initial_balance
        
        # 수수료 및 슬리피지 설정
        self.fee_rate = 0.0005  # 0.05%
        self.slippage = 0.0005  # 0.05%

    def add_trade(self, entry_time: datetime, exit_time: datetime, 
                 entry_price: float, exit_price: float, 
                 position: int, pnl: float):
        """거래 기록 추가"""
        try:
            # 수수료와 슬리피지 계산
            trade_cost = self.trade_amount * (self.fee_rate + self.slippage) * 2  # 진입/청산 각각 발생
            pnl -= trade_cost
            
            # 거래 기록 추가
            trade = {
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': position,
                'pnl': pnl,
                'holding_period': (exit_time - entry_time).total_seconds() / 3600  # 시간 단위
            }
            self.trades.append(trade)
            
            # 수익률 계산
            returns = (pnl / self.trade_amount) * 100
            
            # 승/패 기록
            if pnl > 0:
                self.win_trades += 1
            else:
                self.loss_trades += 1
            
            # 총 수익 업데이트
            self.total_pnl += pnl
            
            # 최대 낙폭 계산
            current_balance = self.initial_balance + self.total_pnl
            if current_balance > self.peak_balance:
                self.peak_balance = current_balance
            current_drawdown = (self.peak_balance - current_balance) / self.peak_balance * 100
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # 거래 로그 출력
            logger.info(f"거래 발생 - 전략: {self.strategy_name}")
            logger.info(f"진입 시간: {entry_time}, 가격: {entry_price:,.2f}")
            logger.info(f"청산 시간: {exit_time}, 가격: {exit_price:,.2f}")
            logger.info(f"포지션: {'롱' if position > 0 else '숏'}, 수익률: {returns:.2f}%")
            logger.info(f"포지션 유지 기간: {trade['holding_period']:.1f}시간")
            logger.info(f"수수료/슬리피지: {trade_cost:,.2f}")
            logger.info("-" * 50)
            
        except Exception as e:
            logger.error(f"거래 기록 추가 중 오류 발생: {str(e)}")
            raise

    def get_statistics(self) -> Dict:
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
            
        total_trades = len(self.trades)
        win_rate = (self.win_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # 일별 수익률 계산
        daily_returns = []
        current_date = None
        daily_pnl = 0
        
        for trade in self.trades:
            trade_date = trade['exit_time'].date()
            if current_date != trade_date:
                if current_date is not None:
                    daily_returns.append(daily_pnl / self.initial_balance)
                current_date = trade_date
                daily_pnl = 0
            daily_pnl += trade['pnl']
        
        if daily_pnl != 0:
            daily_returns.append(daily_pnl / self.initial_balance)
            
        # 샤프 비율 계산
        if daily_returns:
            returns_array = np.array(daily_returns)
            sharpe_ratio = np.sqrt(252) * (returns_array.mean() / returns_array.std()) if returns_array.std() != 0 else 0
        else:
            sharpe_ratio = 0
            
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }

def load_data(file_path: str) -> pd.DataFrame:
    """데이터 로드"""
    try:
        print(f"데이터 파일 경로: {file_path}")
        print(f"파일 존재 여부: {os.path.exists(file_path)}")
        
        print("데이터 파일 읽기 시작...")
        df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
        print("데이터 파일 읽기 완료")
        
        # 불필요한 컬럼 제거
        if 'Unnamed: 0' in df.columns:
            print("불필요한 컬럼 'Unnamed: 0' 제거")
            df = df.drop('Unnamed: 0', axis=1)
        
        # 시간 순서대로 정렬
        df = df.sort_index()
        
        # 2023년부터 2024년까지의 데이터만 필터링
        df = df['2023-01-01':'2024-12-31']
        
        print(f"데이터 크기: {df.shape}")
        print(f"데이터 컬럼: {df.columns.tolist()}")
        print("\n데이터 샘플:")
        print(df.head())
        print("\n데이터 정보:")
        print(df.info())
        
        return df
        
    except Exception as e:
        logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
        raise

def add_trade(trades: list, strategy: str, entry_time: pd.Timestamp, entry_price: float,
             exit_time: pd.Timestamp, exit_price: float, position: str, 
             holding_periods: float, fee: float, initial_balance: float) -> None:
    """거래 기록 추가"""
    try:
        # 수수료 계산
        entry_fee = entry_price * 0.001  # 진입 수수료 0.1%
        exit_fee = exit_price * 0.001    # 청산 수수료 0.1%
        total_fee = entry_fee + exit_fee
        
        # 수익 계산
        if position == '롱':
            profit = (exit_price - entry_price) - total_fee
        else:  # 숏
            profit = (entry_price - exit_price) - total_fee
            
        # 수익률 계산
        returns = (profit / initial_balance) * 100
        
        # 거래 기록 추가
        trades.append({
            'strategy': strategy,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'position': position,
            'profit': profit,
            'returns': returns,
            'holding_periods': holding_periods,
            'fee': total_fee
        })
        
        # 거래 로그 출력
        logger.info(f"거래 발생 - 전략: {strategy}")
        logger.info(f"진입 시간: {entry_time}, 가격: {entry_price:,.2f}")
        logger.info(f"청산 시간: {exit_time}, 가격: {exit_price:,.2f}")
        logger.info(f"포지션: {position}, 수익: {profit:,.0f}원 ({returns:.2f}%)")
        logger.info(f"포지션 유지 기간: {holding_periods}시간")
        logger.info(f"수수료/슬리피지: {total_fee:,.2f}")
        logger.info("-" * 50)
        
    except Exception as e:
        logger.error(f"거래 기록 추가 중 오류 발생: {str(e)}")
        raise

def run_backtest(df: pd.DataFrame, strategy: str, initial_balance: float = 1000000, trade_ratio: float = 0.2) -> BacktestResult:
    """백테스트 실행"""
    try:
        print(f"\n[{strategy}] 백테스트 시작\n")
        
        # 백테스트 결과 객체 생성
        result = BacktestResult(strategy, initial_balance, trade_ratio)
        
        # 전략 객체 생성
        if strategy == 'mean_revert':
            signal_generator = MeanRevertSignal()
        elif strategy == 'ml_boost':
            signal_generator = MLBoostSignal()
        else:
            raise ValueError(f"알 수 없는 전략: {strategy}")
        
        # 신호 생성
        print(f"[{strategy}] 신호 생성 시작")
        print(f"[{strategy}] 입력 데이터 크기: {df.shape}")
        print(f"[{strategy}] 입력 데이터 컬럼: {df.columns.tolist()}")
        
        df = signal_generator._generate_signal(df)
        
        print(f"[{strategy}] 신호 생성 완료")
        print(f"[{strategy}] 출력 데이터 크기: {df.shape}")
        print(f"[{strategy}] 출력 데이터 컬럼: {df.columns.tolist()}")
        
        # 시그널 시프트
        print(f"[{strategy}] 시그널 시프트 시작")
        df['entry_signal'] = df['entry_signal'].shift(1)
        df['exit_signal'] = df['exit_signal'].shift(1)
        print(f"[{strategy}] 시그널 시프트 완료")
        
        # 거래 실행
        print(f"[{strategy}] 거래 실행 시작")
        position = None
        entry_time = None
        entry_price = None
        current_balance = initial_balance
        
        for i in range(len(df)):
            if i % 50 == 0:
                print(f"[{strategy}] 진행중... {i}/{len(df)}")
                
            current_time = df.index[i]
            current_price = df['close'].iloc[i]
            
            # 진입 신호
            if df['entry_signal'].iloc[i] == 1 and position is None:  # 롱 진입
                position = 1
                entry_time = current_time
                entry_price = current_price
            elif df['entry_signal'].iloc[i] == -1 and position is None:  # 숏 진입
                position = -1
                entry_time = current_time
                entry_price = current_price
                
            # 청산 신호
            elif df['exit_signal'].iloc[i] == 1 and position is not None:
                exit_time = current_time
                exit_price = current_price
                
                # 동적 자본 관리: 남은 자본의 20%만 거래
                trade_amount = current_balance * trade_ratio
                if position == 1:  # 롱 포지션
                    pnl = trade_amount * ((exit_price - entry_price) / entry_price)
                else:  # 숏 포지션
                    pnl = trade_amount * ((entry_price - exit_price) / entry_price)
                
                # 거래 기록 추가
                result.add_trade(entry_time, exit_time, entry_price, exit_price, position, pnl)
                
                # 자본 업데이트
                current_balance += pnl
                if current_balance < 0:
                    current_balance = 0
                
                position = None
                entry_time = None
                entry_price = None
        
        print(f"[{strategy}] 거래 실행 완료")
        print(f"[{strategy}] 백테스트 완료")
        
        return result
        
    except Exception as e:
        logger.error(f"백테스트 실행 중 오류 발생: {str(e)}")
        raise

def calculate_metrics(trades: list) -> dict:
    """성과 지표 계산"""
    try:
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
        
        # 기본 지표
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['profit'] > 0])
        win_rate = (winning_trades / total_trades) * 100
        
        # 수익 계산
        total_profit = sum(t['profit'] for t in trades)
        
        # 최대 낙폭 계산
        cumulative_returns = np.cumsum([t['profit'] for t in trades])
        max_drawdown = 0
        peak = cumulative_returns[0]
        
        for ret in cumulative_returns:
            if ret > peak:
                peak = ret
            drawdown = (peak - ret) / peak * 100 if peak != 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # 샤프 비율 계산
        returns = np.array([t['returns'] for t in trades])
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
        
    except Exception as e:
        logger.error(f"성과 지표 계산 중 오류 발생: {str(e)}")
        raise

def main():
    try:
        # 데이터 로드
        data_file = 'data/raw/KRW-BTC_1h.csv'
        df = load_data(data_file)

        # 초기 자본금 설정
        initial_balance = 1000000  # 100만원
        trade_ratio = 0.2  # 각 거래마다 남은 자본의 20% 사용
        
        # 전략별 백테스트 실행
        strategies = ['mean_revert', 'ml_boost']
        results = {}
        
        for strategy in strategies:
            result = run_backtest(df, strategy, initial_balance, trade_ratio)
            results[strategy] = result
        
        # 결과 출력
        print("\n=== 백테스트 결과 ===\n")
        
        for strategy in strategies:
            result = results[strategy]
            stats = result.get_statistics()
            print(f"[{strategy}]")
            print(f"총 거래 수: {stats['total_trades']}")
            print(f"승률: {stats['win_rate']:.1f}%")
            print(f"총 수익: {stats['total_pnl']:,.0f}원")
            print(f"최대 낙폭: {stats['max_drawdown']:.1f}%")
            print(f"샤프 비율: {stats['sharpe_ratio']:.2f}\n")
        
        # 결과 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = project_root / 'data' / 'processed' / 'backtest_results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_data = {
            'timestamp': timestamp,
            'strategies': {}
        }
        
        for strategy in strategies:
            result = results[strategy]
            stats = result.get_statistics()
            results_data['strategies'][strategy] = {
                'trades': result.trades,
                'metrics': stats
            }
        
        results_file = results_dir / f'backtest_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        print(f"백테스트 결과가 저장되었습니다: {results_file}")
        
    except Exception as e:
        logger.error(f"백테스트 실행 중 오류 발생: {str(e)}")
        raise

if __name__ == '__main__':
    main() 