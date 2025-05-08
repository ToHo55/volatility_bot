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
import seaborn as sns # type: ignore
from datetime import datetime, timedelta, time as dt_time
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
import sys
import ta # type: ignore
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
        # 변동성 기반 포지션 크기 조절
        vol_based_size = portfolio_value * (0.2 / volatility) if volatility > 0 else 0
        return min(
            portfolio_value * self.position_size_limit,
            vol_based_size
        )
        
    def should_reduce_exposure(self, current_drawdown: float) -> bool:
        return current_drawdown < self.max_drawdown_limit

    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Value at Risk 계산"""
        return np.percentile(returns, (1 - confidence_level) * 100)

class DataManager:
    def __init__(self):
        self.cache = {}
        self.default_interval = "minute5"
        self.rate_limit_delay = 0.2  # API 호출 간격 (초)
        self.last_request_time = time.time()
        
    def _wait_for_rate_limit(self):
        """API 호출 간격 조절"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
        
    def fetch_ohlcv_with_retry(self, ticker: str, interval: str, count: int, 
                              to: Optional[str] = None, max_retries: int = 3) -> Optional[pd.DataFrame]:
        cache_key = f"{ticker}_{interval}_{count}_{to}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        base_delay = 0.5  # 0.1 -> 0.5
        for attempt in range(max_retries):
            try:
                # API 호출 간격 조절
                self._wait_for_rate_limit()
                
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                
                df = pyupbit.get_ohlcv(ticker, interval=interval, count=count, to=to)
                if df is not None and not df.empty:
                    # 인덱스가 datetime인지 확인
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    
                    # 시간대 설정
                    if df.index.tz is None:
                        df.index = df.index.tz_localize(KST)
                    elif df.index.tz != KST:
                        df.index = df.index.tz_convert(KST)
                    
                    self.cache[cache_key] = df
                    return df
            except Exception as e:
                if "429" in str(e):  # Rate limit error
                    logging.warning(f"[API 호출 제한] {ticker}: 대기 후 재시도")
                    time.sleep(2 ** attempt)  # 지수 백오프
                    continue
                elif attempt == max_retries - 1:
                    logging.warning(f"[데이터 조회 실패] {ticker}: {str(e)}")
                continue
        return None

    def get_time_based_slice(self, df: pd.DataFrame, current_time: datetime, 
                            lookback_minutes: int = 120) -> pd.DataFrame:
        """시간 기반 데이터 슬라이싱"""
        if df is None or df.empty:
            return pd.DataFrame()
            
        # current_time이 datetime 타입이고 시간대 정보가 있는지 확인
        if not isinstance(current_time, datetime):
            current_time = pd.to_datetime(current_time)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=KST)
        elif current_time.tzinfo != KST:
            current_time = current_time.astimezone(KST)
            
        # df의 인덱스가 datetime이고 시간대 정보가 있는지 확인
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize(KST)
        elif df.index.tz != KST:
            df.index = df.index.tz_convert(KST)
            
        end_time = pd.Timestamp(current_time)
        start_time = end_time - pd.Timedelta(minutes=lookback_minutes)
        
        # 디버깅 로그
        logging.debug(f"[데이터 슬라이싱] 시작: {start_time}, 종료: {end_time}")
        logging.debug(f"[데이터 범위] 처음: {df.index[0]}, 마지막: {df.index[-1]}")
        
        mask = (df.index >= start_time) & (df.index <= end_time)
        sliced_df = df[mask]
        
        if sliced_df.empty:
            logging.debug("[데이터 슬라이싱] 결과가 비어있음")
        else:
            logging.debug(f"[데이터 슬라이싱] 결과 크기: {len(sliced_df)}")
            
        return sliced_df

    def resample_to_minute1(self, df: pd.DataFrame) -> pd.DataFrame:
        """5분봉을 1분봉으로 리샘플링"""
        if df is None or df.empty:
            return pd.DataFrame()
            
        # 인덱스가 datetime이고 시간대 정보가 있는지 확인
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize(KST)
        elif df.index.tz != KST:
            df.index = df.index.tz_convert(KST)
            
        # OHLCV 데이터 리샘플링
        resampled = pd.DataFrame()
        resampled['open'] = df['open'].resample('1T').first()
        resampled['high'] = df['high'].resample('1T').max()
        resampled['low'] = df['low'].resample('1T').min()
        resampled['close'] = df['close'].resample('1T').last()
        resampled['volume'] = df['volume'].resample('1T').sum()
        
        # 결측값 처리
        resampled = resampled.fillna(method='ffill')
        
        return resampled

class BaseStrategy(ABC):
    def __init__(self):
        self.risk_manager = RiskManager()
        self.fees = TradingFees()
        self.data_manager = DataManager()
        
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
        """변동성 계산"""
        returns = df['close'].pct_change()
        return returns.std() * np.sqrt(252)

class RSIMAStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        # 진입 조건 완화
        self.volume_ratio_threshold = 0.5
        self.price_increase_threshold = 0.0005
        self.max_holding_time = 120
        self.stop_loss_threshold = -0.015
        self.take_profit_threshold = 0.01
        self.rsi_threshold = 60
        self.ma_slope_threshold = -0.0005
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or len(df) < 20:
            logging.debug("[지표 계산] 충분한 데이터가 없음")
            return pd.DataFrame()
            
        df = df.copy()
        
        try:
            # RSI 계산
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=10).rsi()
            
            # 이동평균선 계산
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma10'] = df['close'].rolling(window=10).mean()
            df['ma5'] = df['close'].rolling(window=5).mean()
            
            # 이동평균선 기울기
            df['ma_slope'] = (df['ma5'] - df['ma5'].shift(1)) / df['ma5'].shift(1)
            
            # 볼린저 밴드
            bb = ta.volatility.BollingerBands(df['close'], window=20)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_mid'] = bb.bollinger_mavg()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            # 디버깅 로그
            logging.debug(f"[지표 계산] 성공 (데이터 크기: {len(df)})")
            logging.debug(f"[지표 값] RSI: {df['rsi'].iloc[-1]:.2f}, "
                        f"MA Slope: {df['ma_slope'].iloc[-1]:.4f}")
            
            return df
            
        except Exception as e:
            logging.error(f"[지표 계산 오류] {str(e)}")
            return pd.DataFrame()
        
    def check_entry_conditions(self, df: pd.DataFrame, current_index: int) -> bool:
        if df is None or df.empty or current_index < 20:
            logging.debug("[진입 조건] 충분한 데이터가 없음")
            return False
            
        try:
            current_data = df.iloc[current_index]
            prev_data = df.iloc[current_index - 1]
            
            # 기본 조건
            volume_ratio = current_data['volume'] / df['volume'].rolling(window=20).mean().iloc[current_index]
            price_increase = (current_data['close'] - prev_data['close']) / prev_data['close']
            
            # RSI 조건
            rsi_condition = (current_data['rsi'] < self.rsi_threshold and 
                           current_data['rsi'] > prev_data['rsi'])
            
            # 이동평균선 조건
            ma_condition = (current_data['ma5'] > current_data['ma10'] and
                          current_data['ma10'] > current_data['ma20'])
            
            # 볼린저 밴드 조건
            bb_condition = (prev_data['close'] <= prev_data['bb_lower'] and
                          current_data['close'] > current_data['bb_lower'])
            
            # MACD 조건
            macd_condition = (current_data['macd'] > current_data['macd_signal'] or
                            (current_data['macd'] > prev_data['macd'] and
                             current_data['macd_signal'] > prev_data['macd_signal']))
            
            # 디버깅 로그
            logging.debug(f"[진입 조건 체크] 거래량 비율: {volume_ratio:.2f}, "
                        f"가격 상승률: {price_increase:.4f}")
            logging.debug(f"[진입 조건 결과] RSI: {rsi_condition}, MA: {ma_condition}, "
                        f"BB: {bb_condition}, MACD: {macd_condition}")
            
            # 종합 조건
            return ((volume_ratio > self.volume_ratio_threshold and 
                    price_increase > self.price_increase_threshold) and
                    (rsi_condition or bb_condition) and
                    (ma_condition or macd_condition))
                    
        except Exception as e:
            logging.error(f"[진입 조건 체크 오류] {str(e)}")
            return False

    def should_exit_position(self, df: pd.DataFrame, entry_price: float, hold_time: float) -> Tuple[bool, str]:
        if len(df) < 2:
            return False, ""
            
        current_data = df.iloc[-1]
        prev_data = df.iloc[-2]
        
        current_price = current_data['close']
        high_price = current_data['high']
        low_price = current_data['low']
        
        # 수익률 계산 (수수료 포함)
        sell_price = self.fees.calculate_sell_price(current_price)
        buy_price = self.fees.calculate_buy_price(entry_price)
        profit_ratio = (sell_price / buy_price) - 1
        
        # 고가/저가 기준 수익률
        high_profit = (self.fees.calculate_sell_price(high_price) / buy_price) - 1
        low_profit = (self.fees.calculate_sell_price(low_price) / buy_price) - 1
        
        # 익절 조건
        if high_profit >= self.take_profit_threshold:
            return True, "익절"
            
        # 손절 조건
        if low_profit <= self.stop_loss_threshold:
            return True, "손절"
            
        # 추가 청산 조건
        
        # 1. RSI 과매수 구간에서 하락 반전
        if current_data['rsi'] > 70 and current_data['rsi'] < prev_data['rsi']:
            return True, "RSI 과매수"
            
        # 2. 볼린저 밴드 상단 터치 후 하락
        if (prev_data['close'] >= prev_data['bb_upper'] and
            current_data['close'] < current_data['bb_upper']):
            return True, "볼린저 상단 이탈"
            
        # 3. MACD 데드크로스
        if (prev_data['macd'] > prev_data['macd_signal'] and
            current_data['macd'] <= current_data['macd_signal']):
            return True, "MACD 데드크로스"
            
        # 4. 이동평균선 하향 돌파
        if (prev_data['close'] > prev_data['ma10'] and
            current_data['close'] <= current_data['ma10'] and
            profit_ratio > 0):
            return True, "MA 하향돌파"
        
        # 시간 초과 조건
        if hold_time >= self.max_holding_time:
            return True, "시간초과" + ("(이익)" if profit_ratio > 0 else "(손실)")
        
        return False, ""

class BacktestEngine:
    def __init__(self, strategy: BaseStrategy, start_date: str, end_date: str):
        self.strategy = strategy
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.portfolio: Dict = {}
        self.trade_log: List = []
        self.daily_pnl: Dict = {}
        
    def run(self):
        logging.info(f"백테스트 시작: {self.start_date.date()} ~ {self.end_date.date()}")
        
        # 거래대금 상위 코인 조회
        tickers = self._get_top_coins(15)
        if not tickers:
            logging.error("분석 대상 코인 조회 실패")
            return
            
        logging.info(f"분석 대상: {', '.join(tickers)} (총 {len(tickers)}개)")
        
        start_time = time.time()
        timeout = 600  # 10분 타임아웃
        
        current_date = self.start_date.date()
        while current_date <= self.end_date.date():
            if time.time() - start_time > timeout:
                logging.error("백테스트 시간 초과 (10분)")
                break
                
            self.daily_pnl[current_date] = 0.0
            logging.info(f"\n=== {current_date} 백테스트 진행 중 ({(current_date - self.start_date.date()).days + 1}일차) ===")
            
            # 병렬 처리 대신 순차 처리로 변경
            ohlcv_cache = {}
            for ticker in tickers:
                df = self._fetch_daily_data(ticker, current_date)
                if df is not None:
                    ohlcv_cache[ticker] = df
                time.sleep(0.2)  # API 호출 간격 조절
            
            # 시간대별 처리
            self._process_timeframe(current_date, ohlcv_cache, tickers)
            
            if current_date in self.daily_pnl:
                logging.info(f"\n[{current_date} 거래 완료] 일간 수익률: {self.daily_pnl[current_date]*100:.2f}%")
            
            current_date += timedelta(days=1)
            
        return pd.DataFrame(self.trade_log)

    def _get_top_coins(self, limit: int = 15) -> List[str]:
        """거래대금 상위 코인 조회"""
        try:
            tickers = pyupbit.get_tickers(fiat="KRW")
            time.sleep(0.2)  # API 호출 간격 조절
            
            all_day_candles = []
            for ticker in tickers:
                try:
                    time.sleep(0.2)  # API 호출 간격 조절
                    df = pyupbit.get_ohlcv(ticker, interval="day", count=1)
                    if df is not None:
                        volume_krw = df['close'].iloc[-1] * df['volume'].iloc[-1]
                        all_day_candles.append((ticker, volume_krw))
                except Exception as e:
                    logging.warning(f"{ticker} 데이터 조회 실패")
                    continue
            
            sorted_tickers = sorted(all_day_candles, key=lambda x: x[1], reverse=True)
            return [t[0] for t in sorted_tickers[:limit]]
        except Exception as e:
            logging.error("거래대금 상위 코인 조회 실패")
            return []

    def _fetch_daily_data(self, ticker: str, current_date: datetime.date) -> Optional[pd.DataFrame]:
        """일별 데이터 수집"""
        next_date = current_date + timedelta(days=1)
        return self.strategy.data_manager.fetch_ohlcv_with_retry(
            ticker, "minute5", 288,  # 5분봉 기준 하루 288개
            to=next_date.strftime("%Y-%m-%d")
        )

    def _process_timeframe(self, current_date: datetime.date, ohlcv_cache: Dict, tickers: List[str]):
        """시간대별 처리"""
        for hour in range(9, 21):
            for minute in range(0, 60, 5):  # 5분 단위로 처리
                current_dt = datetime.combine(current_date, 
                                           dt_time(hour, minute)).replace(tzinfo=KST)
                
                if minute == 0:
                    self._log_portfolio_status(current_dt, ohlcv_cache)
                
                self._execute_trading_logic(current_dt, ohlcv_cache, tickers)

    def _log_portfolio_status(self, current_dt: datetime, ohlcv_cache: Dict):
        """포트폴리오 상태 로깅"""
        if self.portfolio:
            logging.info(f"\n[{current_dt.strftime('%H:%M')}] 포트폴리오 상태")
            for t, (p, tm) in self.portfolio.items():
                if t in ohlcv_cache:
                    df = ohlcv_cache[t]
                    current_data = self.strategy.data_manager.get_time_based_slice(
                        df, current_dt)
                    if not current_data.empty:
                        hold_time = (current_dt - tm).total_seconds() / 60
                        current_price = current_data['close'].iloc[-1]
                        profit = ((self.strategy.fees.calculate_sell_price(current_price)) / 
                                (self.strategy.fees.calculate_buy_price(p))) - 1
                        logging.info(f"- {t}: 진입가 {p:,.0f}원 / 수익률 {profit*100:.1f}% / "
                                   f"보유시간 {hold_time:.0f}분")

    def _execute_trading_logic(self, current_dt: datetime, ohlcv_cache: Dict, tickers: List[str]):
        """매매 로직 실행"""
        current_date = current_dt.date()
        
        logging.info(f"\n[{current_dt.strftime('%Y-%m-%d %H:%M')}] 매매 로직 실행")
        logging.info(f"- 현재 포트폴리오: {list(self.portfolio.keys())}")
        
        # 매도 로직
        for ticker in list(self.portfolio.keys()):
            try:
                buy_price, buy_time = self.portfolio[ticker]
                if ticker in ohlcv_cache:
                    df = ohlcv_cache[ticker]
                    current_data = self.strategy.data_manager.get_time_based_slice(
                        df, current_dt)
                    
                    if current_data.empty:
                        logging.debug(f"[{ticker}] 현재 데이터 없음")
                        continue
                    
                    current_price = current_data['close'].iloc[-1]    
                    hold_time = (current_dt - buy_time).total_seconds() / 60
                    
                    logging.debug(f"[{ticker}] 현재가: {current_price:,.0f}원, "
                                f"매수가: {buy_price:,.0f}원, "
                                f"보유시간: {hold_time:.1f}분")
                    
                    should_sell, reason = self.strategy.should_exit_position(
                        current_data, buy_price, hold_time)
                    
                    if should_sell:
                        profit_ratio = ((self.strategy.fees.calculate_sell_price(current_price)) / 
                                      (self.strategy.fees.calculate_buy_price(buy_price))) - 1
                        
                        self.daily_pnl[current_date] += profit_ratio
                        
                        self.trade_log.append({
                            "ticker": ticker,
                            "type": "sell",
                            "price": current_price,
                            "time": current_dt,
                            "pnl": profit_ratio,
                            "reason": reason
                        })
                        
                        logging.info(f"[매도 실행] {ticker}")
                        logging.info(f"- 매수가: {buy_price:,.0f}원")
                        logging.info(f"- 매도가: {current_price:,.0f}원")
                        logging.info(f"- 수익률: {profit_ratio*100:.2f}%")
                        logging.info(f"- 청산사유: {reason}")
                        logging.info(f"- 보유시간: {hold_time:.1f}분")
                        
                        del self.portfolio[ticker]
            except Exception as e:
                logging.error(f"[매도 오류] {ticker}: {str(e)}")
                continue

        # 매수 로직
        for ticker in tickers:
            if ticker not in self.portfolio and len(self.portfolio) < 3:
                try:
                    if ticker in ohlcv_cache:
                        df = ohlcv_cache[ticker].copy()
                        current_data = self.strategy.data_manager.get_time_based_slice(
                            df, current_dt)
                        
                        if len(current_data) < 20:
                            logging.debug(f"[{ticker}] 충분한 데이터 없음")
                            continue
                            
                        # 지표 계산
                        df_indicators = self.strategy.calculate_indicators(current_data)
                        
                        if df_indicators.empty:
                            logging.debug(f"[{ticker}] 지표 계산 실패")
                            continue
                            
                        current_price = df_indicators['close'].iloc[-1]
                        volume_ratio = df_indicators['volume'].iloc[-1] / df_indicators['volume'].rolling(window=20).mean().iloc[-1]
                        
                        logging.debug(f"[{ticker}] 진입 조건 체크")
                        logging.debug(f"- 현재가: {current_price:,.0f}원")
                        logging.debug(f"- 거래량 비율: {volume_ratio:.2f}")
                        if 'rsi' in df_indicators.columns:
                            logging.debug(f"- RSI: {df_indicators['rsi'].iloc[-1]:.1f}")
                        
                        # 진입 조건 확인
                        if self.strategy.check_entry_conditions(df_indicators, -1):
                            entry_price = df_indicators['close'].iloc[-1]
                            
                            # 변동성 기반 포지션 크기 계산
                            volatility = self.strategy.calculate_volatility(df_indicators)
                            position_size = self.strategy.risk_manager.calculate_position_size(
                                volatility, 1000000)  # 예시 포트폴리오 크기
                            
                            self.portfolio[ticker] = (entry_price, current_dt)
                            self.trade_log.append({
                                "ticker": ticker,
                                "type": "buy",
                                "price": entry_price,
                                "time": current_dt,
                                "position_size": position_size
                            })
                            
                            logging.info(f"[매수 실행] {ticker}")
                            logging.info(f"- 매수가: {entry_price:,.0f}원")
                            logging.info(f"- 포지션 크기: {position_size:,.0f}원")
                            logging.info(f"- 변동성: {volatility:.4f}")
                except Exception as e:
                    logging.error(f"[매수 오류] {ticker}: {str(e)}")
                    continue

class BacktestAnalyzer:
    def __init__(self, trade_log_df):
        self.df = trade_log_df
        self.prepare_data()

    def prepare_data(self):
        """거래 데이터 전처리"""
        if len(self.df) == 0:
            logging.warning("거래 기록이 없습니다.")
            self.buy_trades = pd.DataFrame()
            self.sell_trades = pd.DataFrame()
            return
            
        time_column = 'time_buy' if 'time_buy' in self.df.columns else 'time'
        self.df['date'] = pd.to_datetime(self.df[time_column]).dt.date
        
        self.buy_trades = self.df[self.df['type'] == 'buy']
        self.sell_trades = self.df[self.df['type'] == 'sell']
        
        # 포지션 크기 정규화
        if 'position_size' in self.buy_trades.columns:
            total_capital = self.buy_trades['position_size'].sum()
            self.buy_trades['weight'] = self.buy_trades['position_size'] / total_capital

    def calculate_metrics(self):
        """주요 성과 지표 계산"""
        if len(self.df) == 0:
            return self._get_empty_metrics()
            
        metrics = {
            'total_trades': len(self.sell_trades),
            'winning_trades': len(self.sell_trades[self.sell_trades['pnl'] > 0]),
            'losing_trades': len(self.sell_trades[self.sell_trades['pnl'] <= 0]),
            'total_pnl': self.sell_trades['pnl'].sum() * 100,
            'avg_pnl': self.sell_trades['pnl'].mean() * 100,
            'max_drawdown': self.calculate_max_drawdown(),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'sortino_ratio': self.calculate_sortino_ratio(),
            'calmar_ratio': self.calculate_calmar_ratio(),
            'avg_hold_time': self.calculate_avg_hold_time(),
            'var_95': self.calculate_var(0.95),
            'var_99': self.calculate_var(0.99)
        }
        
        metrics['win_rate'] = (metrics['winning_trades'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0
        return metrics

    def _get_empty_metrics(self):
        """빈 메트릭스 반환"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'avg_hold_time': 0.0,
            'var_95': 0.0,
            'var_99': 0.0
        }

    def calculate_max_drawdown(self):
        """최대 낙폭 계산"""
        if len(self.sell_trades) == 0:
            return 0.0
            
        # 누적 수익률 계산 (포지션 크기 가중치 적용)
        if 'weight' in self.sell_trades.columns:
            weighted_returns = self.sell_trades['pnl'] * self.sell_trades['weight']
            cumulative = (1 + weighted_returns).cumprod()
        else:
            cumulative = (1 + self.sell_trades['pnl']).cumprod()
            
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        return drawdowns.min() * 100

    def calculate_sharpe_ratio(self):
        """샤프 비율 계산"""
        if len(self.sell_trades) == 0:
            return 0.0
            
        # 일간 수익률 계산 (포지션 크기 가중치 적용)
        if 'weight' in self.sell_trades.columns:
            daily_returns = self.sell_trades.groupby('date').apply(
                lambda x: (x['pnl'] * x['weight']).sum())
        else:
            daily_returns = self.sell_trades.groupby('date')['pnl'].sum()
            
        if len(daily_returns) > 0:
            return np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() != 0 else 0
        return 0

    def calculate_sortino_ratio(self):
        """소르티노 비율 계산"""
        if len(self.sell_trades) == 0:
            return 0.0
            
        # 일간 수익률
        daily_returns = self.sell_trades.groupby('date')['pnl'].sum()
        
        if len(daily_returns) > 0:
            # 하방 표준편차 계산
            negative_returns = daily_returns[daily_returns < 0]
            downside_std = np.sqrt(np.mean(negative_returns**2)) if len(negative_returns) > 0 else 0
            
            if downside_std != 0:
                return np.sqrt(252) * (daily_returns.mean() / downside_std)
        return 0

    def calculate_calmar_ratio(self):
        """칼마 비율 계산"""
        if len(self.sell_trades) == 0:
            return 0.0
            
        max_dd = self.calculate_max_drawdown()
        if max_dd != 0:
            annual_return = self.sell_trades['pnl'].mean() * 252 * 100
            return annual_return / abs(max_dd)
        return 0

    def calculate_var(self, confidence_level=0.95):
        """Value at Risk 계산"""
        if len(self.sell_trades) == 0:
            return 0.0
            
        # 포지션 크기 가중치 적용
        if 'weight' in self.sell_trades.columns:
            returns = self.sell_trades['pnl'] * self.sell_trades['weight']
        else:
            returns = self.sell_trades['pnl']
            
        return np.percentile(returns, (1 - confidence_level) * 100) * 100

    def calculate_avg_hold_time(self):
        """평균 보유 시간 계산 (분)"""
        if len(self.sell_trades) == 0:
            return 0.0
        merged = pd.merge(self.buy_trades, self.sell_trades, on='ticker', suffixes=('_buy', '_sell'))
        merged['hold_time'] = (pd.to_datetime(merged['time_sell']) - pd.to_datetime(merged['time_buy'])).dt.total_seconds() / 60
        return merged['hold_time'].mean()

    def save_analysis_log(self):
        """상세 분석 결과를 로그 파일로 저장"""
        if len(self.df) == 0:
            logging.info("\n=== 백테스트 분석 결과 ===")
            logging.info("거래 기록이 없습니다.")
            return

        # 기본 메트릭스 계산
        metrics = self.calculate_metrics()
        
        # 분석 결과 로깅
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
        logging.info(f"- VaR(95%): {metrics['var_95']:.2f}%")
        logging.info(f"- VaR(99%): {metrics['var_99']:.2f}%")
        
        logging.info(f"\n3. 위험 조정 수익률")
        logging.info(f"- 샤프 비율: {metrics['sharpe_ratio']:.2f}")
        logging.info(f"- 소르티노 비율: {metrics['sortino_ratio']:.2f}")
        logging.info(f"- 칼마 비율: {metrics['calmar_ratio']:.2f}")
        
        logging.info(f"\n4. 거래 시간 분석")
        logging.info(f"- 평균 보유 시간: {metrics['avg_hold_time']:.1f}분")
        
        if len(self.sell_trades) > 0:
            logging.info(f"\n5. 수익률 분포")
            pnl_stats = self.sell_trades['pnl'].multiply(100).describe()
            logging.info(f"- 최소: {pnl_stats['min']:.2f}%")
            logging.info(f"- 1분위: {pnl_stats['25%']:.2f}%")
            logging.info(f"- 중앙값: {pnl_stats['50%']:.2f}%")
            logging.info(f"- 3분위: {pnl_stats['75%']:.2f}%")
            logging.info(f"- 최대: {pnl_stats['max']:.2f}%")
            
            logging.info(f"\n6. 종목별 분석")
            by_ticker = self.sell_trades.groupby('ticker').agg({
                'pnl': ['count', 'mean', 'sum']
            }).round(4)
            by_ticker.columns = ['거래횟수', '평균수익률', '총수익률']
            for ticker, row in by_ticker.iterrows():
                logging.info(f"\n{ticker}:")
                logging.info(f"- 거래횟수: {row['거래횟수']}회")
                logging.info(f"- 평균수익률: {row['평균수익률']*100:.2f}%")
                logging.info(f"- 총수익률: {row['총수익률']*100:.2f}%")

def main():
    logging.info("백테스트 시작")
    
    start_date = "2025-04-13"
    end_date = "2025-04-15"
    
    strategy = RSIMAStrategy()
    engine = BacktestEngine(strategy, start_date, end_date)
    
    df_result = engine.run()
    if df_result is not None and not df_result.empty:
        df_result.to_csv("backtest_result.csv", index=False, encoding="utf-8-sig")
        
        analyzer = BacktestAnalyzer(df_result)
        analyzer.save_analysis_log()

if __name__ == "__main__":
    main()
