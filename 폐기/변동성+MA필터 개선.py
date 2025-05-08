#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import logging
import pyupbit
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from datetime import datetime, timedelta, time as dt_time
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
import sys
import ta  # type: ignore
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 환경 설정
load_dotenv()
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

KST = ZoneInfo("Asia/Seoul")

# 로깅 설정
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/backtest.log", encoding='utf-8'),
        logging.FileHandler("logs/backtest_analysis.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

class TradingFees:
    def __init__(self):
        self.buy_slippage = 0.002  # 매수 슬리피지 0.2%
        self.sell_slippage = 0.001  # 매도 슬리피지 0.1%
        self.buy_fee = 0.0015      # 매수 수수료 0.15%
        self.sell_fee = 0.0025     # 매도 수수료 0.25%

    def calculate_buy_price(self, price: float) -> float:
        return price * (1 + self.buy_slippage + self.buy_fee)

    def calculate_sell_price(self, price: float) -> float:
        return price * (1 - self.sell_slippage - self.sell_fee)

class RiskManager:
    def __init__(self):
        self.max_daily_loss = -0.05
        self.max_drawdown_limit = -0.15
        self.var_confidence = 0.95
        self.position_size_limit = 0.1  # 기본 최대 포지션 크기 (10%)
        
    def calculate_position_size(self, volatility: float, portfolio_value: float) -> float:
        vol_based_size = portfolio_value * (0.2 / volatility) if volatility > 0 else 0
        return min(portfolio_value * self.position_size_limit, vol_based_size)

class BaseStrategy(ABC):
    def __init__(self):
        self.risk_manager = RiskManager()
        self.fees = TradingFees()
        
    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
        
    @abstractmethod
    def check_entry_conditions(self, df: pd.DataFrame, current_index: int) -> bool:
        pass
        
    @abstractmethod
    def should_exit_position(self, df: pd.DataFrame, entry_price: float, hold_time: float) -> Tuple[bool, str]:
        pass

    def calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> float:
        returns = df['close'].pct_change()
        return returns.std() * np.sqrt(252)

# 새로운 전략 클래스 정의
class VolatilityMAStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.k = 0.5
        self.ma_period = 10
        self.take_profit = 0.02
        self.stop_loss = -0.015
        self.max_holding_time = 60

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df['high_low_range'] = df['high'] - df['low']
        df['target'] = df['open'] + df['high_low_range'].shift(1) * self.k
        df['ma'] = df['close'].rolling(window=self.ma_period).mean()
        return df

    def check_entry_conditions(self, df: pd.DataFrame, current_index: int) -> bool:
        if len(df) < current_index + 1:
            return False

        row = df.iloc[current_index]
        # 진입 조건: 현재가 > 돌파 가격, 현재가 > 이동평균선
        if row['close'] > row['target'] and row['close'] > row['ma']:
            return True
        return False

    def should_exit_position(self, df: pd.DataFrame, entry_price: float, hold_time: float) -> Tuple[bool, str]:
        if len(df) < 1:
            return False, ""
        current_price = df.iloc[-1]['close']
        sell_price = self.fees.calculate_sell_price(current_price)
        buy_price = self.fees.calculate_buy_price(entry_price)
        profit_ratio = (sell_price / buy_price) - 1

        if profit_ratio >= self.take_profit:
            return True, "익절"
        elif profit_ratio <= self.stop_loss:
            return True, "손절"
        elif hold_time >= self.max_holding_time:
            return True, "시간초과"

        return False, ""

class BacktestEngine:
    def __init__(self, strategy: BaseStrategy, start_date: str, end_date: str):
        self.strategy = strategy
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.portfolio: Dict = {}
        self.trade_log: List = []
        self.daily_pnl: Dict = {}

    def _get_ohlcv(self, ticker: str, current_date: datetime.date) -> Optional[pd.DataFrame]:
        try:
            df = pyupbit.get_ohlcv(ticker, interval="minute5", count=288,
                                  to=datetime.combine(current_date + timedelta(days=1),
                                                    dt_time(9, 0)).strftime("%Y-%m-%d %H:%M:%S"))
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logging.error(f"데이터 조회 실패 ({ticker}): {str(e)}")
        return None

    def run(self):
        logging.info(f"백테스트 시작: {self.start_date.date()} ~ {self.end_date.date()}")
        
        current_date = self.start_date.date()
        while current_date <= self.end_date.date():
            self.daily_pnl[current_date] = 0.0
            logging.info(f"\n=== {current_date} 백테스트 진행 중 ===")
            
            # 거래대금 상위 코인 조회
            tickers = pyupbit.get_tickers(fiat="KRW")[:15]
            
            for ticker in tickers:
                df = self._get_ohlcv(ticker, current_date)
                if df is not None:
                    df = self.strategy.calculate_indicators(df)
                    
                    # 매매 로직 실행
                    for i in range(len(df)):
                        if ticker not in self.portfolio and self.strategy.check_entry_conditions(df, i):
                            entry_price = df.iloc[i]['close']
                            self.portfolio[ticker] = (entry_price, datetime.now())
                            self.trade_log.append({
                                "ticker": ticker,
                                "type": "buy",
                                "price": entry_price,
                                "time": df.index[i]
                            })
                            logging.info(f"매수: {ticker} @ {entry_price:,.0f}")
                        
                        elif ticker in self.portfolio:
                            entry_price, entry_time = self.portfolio[ticker]
                            hold_time = (df.index[i] - entry_time).total_seconds() / 60
                            
                            should_sell, reason = self.strategy.should_exit_position(
                                df.iloc[max(0, i-10):i+1], entry_price, hold_time)
                            
                            if should_sell:
                                exit_price = df.iloc[i]['close']
                                profit_ratio = ((self.strategy.fees.calculate_sell_price(exit_price)) /
                                              (self.strategy.fees.calculate_buy_price(entry_price))) - 1
                                
                                self.daily_pnl[current_date] += profit_ratio
                                
                                self.trade_log.append({
                                    "ticker": ticker,
                                    "type": "sell",
                                    "price": exit_price,
                                    "time": df.index[i],
                                    "pnl": profit_ratio,
                                    "reason": reason
                                })
                                
                                logging.info(f"매도: {ticker} @ {exit_price:,.0f} ({reason}, {profit_ratio*100:.1f}%)")
                                del self.portfolio[ticker]
                
                time.sleep(0.1)  # API 호출 제한 방지
            
            if current_date in self.daily_pnl:
                logging.info(f"\n[{current_date} 거래 완료] 일간 수익률: {self.daily_pnl[current_date]*100:.2f}%")
            
            current_date += timedelta(days=1)
            
        return pd.DataFrame(self.trade_log)

class BacktestAnalyzer:
    def __init__(self, trade_log_df: pd.DataFrame):
        self.df = trade_log_df
        self.prepare_data()

    def prepare_data(self):
        """거래 데이터 전처리"""
        if len(self.df) == 0:
            logging.warning("거래 기록이 없습니다.")
            return
            
        self.df['date'] = pd.to_datetime(self.df['time']).dt.date
        self.buy_trades = self.df[self.df['type'] == 'buy']
        self.sell_trades = self.df[self.df['type'] == 'sell']

    def calculate_metrics(self):
        """주요 성과 지표 계산"""
        if len(self.df) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
            
        metrics = {}
        
        # 기본 거래 통계
        metrics['total_trades'] = len(self.sell_trades)
        metrics['winning_trades'] = len(self.sell_trades[self.sell_trades['pnl'] > 0])
        metrics['losing_trades'] = len(self.sell_trades[self.sell_trades['pnl'] <= 0])
        metrics['win_rate'] = (metrics['winning_trades'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0
        
        # 수익률 분석
        metrics['total_pnl'] = self.sell_trades['pnl'].sum() * 100
        metrics['avg_pnl'] = self.sell_trades['pnl'].mean() * 100
        metrics['max_drawdown'] = self.calculate_max_drawdown()
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio()
        
        return metrics

    def calculate_max_drawdown(self) -> float:
        """최대 낙폭 계산"""
        if len(self.sell_trades) == 0:
            return 0.0
            
        cumulative = (1 + self.sell_trades['pnl']).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        return drawdowns.min() * 100

    def calculate_sharpe_ratio(self) -> float:
        """샤프 비율 계산"""
        if len(self.sell_trades) == 0:
            return 0.0
            
        daily_returns = self.sell_trades.groupby('date')['pnl'].sum()
        if len(daily_returns) > 0:
            return np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() != 0 else 0
        return 0

    def save_analysis_log(self):
        """분석 결과를 로그 파일로 저장"""
        metrics = self.calculate_metrics()
        
        logging.info("\n=== 백테스트 분석 결과 ===")
        logging.info(f"1. 기본 거래 통계")
        logging.info(f"- 총 거래 횟수: {metrics['total_trades']}회")
        logging.info(f"- 승리 거래: {metrics['winning_trades']}회")
        logging.info(f"- 손실 거래: {metrics['losing_trades']}회")
        logging.info(f"- 승률: {metrics['win_rate']:.2f}%")
        
        logging.info(f"\n2. 수익률 분석")
        logging.info(f"- 총 수익률: {metrics['total_pnl']:.2f}%")
        logging.info(f"- 평균 수익률: {metrics['avg_pnl']:.2f}%")
        logging.info(f"- 최대 낙폭: {metrics['max_drawdown']:.2f}%")
        logging.info(f"- 샤프 비율: {metrics['sharpe_ratio']:.2f}")
        
        if len(self.sell_trades) > 0:
            logging.info(f"\n3. 종목별 분석")
            by_ticker = self.sell_trades.groupby('ticker').agg({
                'pnl': ['count', 'mean', 'sum']
            }).round(4)
            by_ticker.columns = ['거래횟수', '평균수익률', '총수익률']
            for ticker, row in by_ticker.iterrows():
                logging.info(f"\n{ticker}:")
                logging.info(f"- 거래횟수: {row['거래횟수']}회")
                logging.info(f"- 평균수익률: {row['평균수익률']*100:.2f}%")
                logging.info(f"- 총수익률: {row['총수익률']*100:.2f}%")

# main 함수 내부에서 전략 교체 가능하도록 수정
if __name__ == "__main__":
    logging.info("백테스트 시작")

    start_date = "2025-04-13"
    end_date = "2025-04-15"

    strategy = VolatilityMAStrategy()  # <- 전략 교체
    engine = BacktestEngine(strategy, start_date, end_date)

    df_result = engine.run()
    if df_result is not None and not df_result.empty:
        df_result.to_csv("backtest_result.csv", index=False, encoding="utf-8-sig")
        
        analyzer = BacktestAnalyzer(df_result)
        analyzer.save_analysis_log()
