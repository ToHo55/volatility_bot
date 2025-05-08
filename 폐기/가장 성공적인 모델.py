# RSI 역추세 전략 자동 백테스트 (다중 알트코인)
# 주요 구성: 전략 클래스 + 다중 티커 루프 + 성과 비교 분석

import os
import time
import logging
import pyupbit
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# 기본 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
os.makedirs("logs", exist_ok=True)

def check_ticker_validity(ticker: str) -> bool:
    """티커의 유효성을 검사합니다."""
    try:
        # 현재가 조회로 유효성 체크
        current_price = pyupbit.get_current_price(ticker)
        return current_price is not None
    except Exception as e:
        logging.error(f"티커 유효성 검사 실패 ({ticker}): {str(e)}")
        return False

def safe_request_with_retry(func, *args, max_retries: int = 5, delay: float = 1.0, **kwargs) -> Optional[pd.DataFrame]:
    """안전한 API 요청 처리를 위한 래퍼 함수"""
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            if result is not None:
                return result
        except Exception as e:
            logging.warning(f"요청 실패 (시도 {attempt + 1}/{max_retries}): {str(e)}")
        
        # 지수 백오프
        time.sleep(delay * (2 ** attempt))
    return None

class RSIReversalStrategy:
    def __init__(self, ticker: str):
        # 코인별 특성
        self.ticker = ticker
        self.is_btc = ticker == "KRW-BTC"
        self.is_major = ticker in ["KRW-BTC", "KRW-XRP", "KRW-SOL"]
        self.is_doge = ticker == "KRW-DOGE"
        
        # RSI 설정 (코인별 차별화)
        self.rsi_period = 14
        if self.is_btc:
            self.entry_threshold = 35    # BTC는 약한 과매도에서도 진입
            self.exit_threshold = 52     # BTC는 중간 정도에서 청산
        elif self.is_doge:
            self.entry_threshold = 30    # DOGE 진입 기준 완화
            self.exit_threshold = 60     # DOGE 청산 기준 완화
        elif self.is_major:
            self.entry_threshold = 25
            self.exit_threshold = 55
        else:
            self.entry_threshold = 20
            self.exit_threshold = 60
        
        # 손익 설정 (코인별 차별화)
        if self.is_btc:
            self.stop_loss = -0.005     # BTC는 매우 타이트한 손절
            self.take_profit = 0.008    # BTC는 작은 수익 실현
            self.trailing_stop = 0.002   # BTC는 매우 좁은 트레일링
        elif self.is_doge:
            self.stop_loss = -0.015     # DOGE 손절 기준 타이트하게 조정
            self.take_profit = 0.025    # DOGE 익절 기준 현실적으로 조정
            self.trailing_stop = 0.008   # DOGE 트레일링 스탑 조정
        elif self.is_major:
            self.stop_loss = -0.015
            self.take_profit = 0.025
            self.trailing_stop = 0.005
        else:
            self.stop_loss = -0.012
            self.take_profit = 0.02
            self.trailing_stop = 0.007
        
        # 시간 설정 (코인별 차별화)
        if self.is_btc:
            self.min_hold_time = 5      # BTC는 매우 짧은 보유
            self.max_hold_time = 30     # BTC는 빠른 회전
        elif self.is_doge:
            self.min_hold_time = 30
            self.max_hold_time = 180
        else:
            self.min_hold_time = 15
            self.max_hold_time = 120
        
        # 변동성 필터 설정
        self.atr_period = 14
        self.vol_window = 20
        if self.is_btc:
            self.min_volatility = 0.002  # BTC는 매우 낮은 변동성에서도 거래
            self.vol_percentile = 40     # BTC는 낮은 변동성 기준
        elif self.is_doge:
            self.min_volatility = 0.004  # DOGE 변동성 기준 완화
            self.vol_percentile = 60     # DOGE 변동성 백분위 완화
        else:
            self.min_volatility = 0.005
            self.vol_percentile = 50
        
        # 거래량 필터
        self.volume_ma_period = 20
        if self.is_btc:
            self.min_volume_ratio = 1.2  # BTC는 낮은 거래량에서도 거래
        elif self.is_doge:
            self.min_volume_ratio = 1.5  # DOGE 거래량 기준 완화
        else:
            self.min_volume_ratio = 2.0
        
        # 추세 필터 설정
        if self.is_btc:
            self.ma_short = 5           # BTC는 더 짧은 이평선
            self.ma_mid = 15
            self.ma_long = 30
        else:
            self.ma_short = 10
            self.ma_mid = 30
            self.ma_long = 60
            
        self.trend_threshold = 0.0005 if self.is_btc else 0.002  # BTC는 매우 작은 추세도 포착
        
        # 연속 손실 제한
        self.max_consecutive_losses = 2
        self.current_consecutive_losses = 0
        
        # 일일 거래 제한
        if self.is_btc:
            self.max_daily_trades = 8    # BTC는 더 많은 거래 허용
        elif self.is_doge:
            self.max_daily_trades = 5    # DOGE 일일 거래 제한 증가
        else:
            self.max_daily_trades = 3
        self.daily_trades = {}
        
        # BTC 전용 추가 필터
        if self.is_btc:
            self.vwap_period = 30       # VWAP 기간
            self.vwap_threshold = 0.0005 # VWAP과의 괴리율

    def can_trade_today(self, current_time) -> bool:
        """일일 거래 제한 확인"""
        date_str = current_time.strftime('%Y-%m-%d')
        if date_str not in self.daily_trades:
            self.daily_trades = {date_str: 0}  # 새로운 날짜면 초기화
        return self.daily_trades[date_str] < self.max_daily_trades

    def update_trade_result(self, profit_ratio: float, current_time) -> None:
        """거래 결과 업데이트"""
        # 연속 손실 카운트 업데이트
        if profit_ratio < 0:
            self.current_consecutive_losses += 1
        else:
            self.current_consecutive_losses = 0
            
        # 일일 거래 횟수 업데이트
        date_str = current_time.strftime('%Y-%m-%d')
        self.daily_trades[date_str] = self.daily_trades.get(date_str, 0) + 1

    def calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """ATR(Average True Range) 계산"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=self.atr_period).mean()

    def is_high_volatility(self, df: pd.DataFrame) -> bool:
        """현재 변동성이 높은지 확인"""
        if len(df) < max(self.vol_window, self.atr_period):
            return False

        # ATR 기반 변동성
        atr = self.calculate_atr(df)
        current_atr = atr.iloc[-1]
        price = df['close'].iloc[-1]
        atr_ratio = current_atr / price  # 가격 대비 ATR 비율

        # 변동성 백분위 계산 (단순화)
        volatility = (df['high'] / df['low'] - 1).rolling(window=self.vol_window).mean()
        vol_threshold = volatility.quantile(self.vol_percentile/100)
        
        # 둘 중 하나만 만족해도 됨 (AND → OR)
        return (atr_ratio > self.min_volatility or 
                volatility.iloc[-1] > vol_threshold)

    def is_high_volume(self, df: pd.DataFrame) -> bool:
        """거래량이 충분한지 확인"""
        if len(df) < self.volume_ma_period:
            return False
            
        volume = df['volume']
        volume_ma = volume.rolling(window=self.volume_ma_period).mean()
        return volume.iloc[-1] > volume_ma.iloc[-1] * self.min_volume_ratio

    def calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """추세 강도 계산"""
        ma_short = df['close'].rolling(window=self.ma_short).mean()
        ma_mid = df['close'].rolling(window=self.ma_mid).mean()
        ma_long = df['close'].rolling(window=self.ma_long).mean()
        
        # 이동평균선 기울기 계산
        short_slope = (ma_short.iloc[-1] / ma_short.iloc[-5] - 1)
        mid_slope = (ma_mid.iloc[-1] / ma_mid.iloc[-10] - 1)
        long_slope = (ma_long.iloc[-1] / ma_long.iloc[-20] - 1)
        
        return (short_slope + mid_slope + long_slope) / 3

    def adjust_parameters(self, df: pd.DataFrame) -> None:
        """시장 상황에 따른 동적 파라미터 조정"""
        trend_strength = self.calculate_trend_strength(df)
        volatility = self.calculate_current_volatility(df)
        
        # 강한 하락 추세
        if trend_strength < -self.trend_threshold:
            self.entry_threshold = 20  # RSI 기준 강화
            self.take_profit = 0.02    # 목표 수익 낮춤
            self.max_hold_time = 90    # 보유 시간 단축
        
        # 강한 상승 추세
        elif trend_strength > self.trend_threshold:
            self.entry_threshold = 30  # RSI 기준 완화
            self.take_profit = 0.03    # 목표 수익 높임
            self.max_hold_time = 150   # 보유 시간 연장
        
        # 높은 변동성 구간
        if volatility > self.min_volatility * 2:
            self.stop_loss = -0.02     # 손절 기준 완화
            self.trailing_stop = 0.007  # 트레일링 스탑 강화
        else:
            self.stop_loss = -0.015    # 기본 손절 기준
            self.trailing_stop = 0.005  # 기본 트레일링 스탑

    def calculate_current_volatility(self, df: pd.DataFrame) -> float:
        """현재 변동성 수준 계산"""
        atr = self.calculate_atr(df)
        return atr.iloc[-1] / df['close'].iloc[-1]

    def is_volatility_increasing(self, df: pd.DataFrame) -> bool:
        """DOGE 전용: 변동성 증가 확인"""
        if not self.is_doge:
            return True
            
        volatility = (df['high'] / df['low'] - 1)
        vol_ma = volatility.rolling(window=self.volatility_ma).mean()
        current_vol = volatility.iloc[-1]
        return current_vol > vol_ma.iloc[-1] * self.vol_increase_threshold

    def calculate_vwap(self, df: pd.DataFrame) -> float:
        """VWAP(Volume Weighted Average Price) 계산"""
        if not self.is_btc:
            return 0
            
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(window=self.vwap_period).sum() / df['volume'].rolling(window=self.vwap_period).sum()
        return vwap.iloc[-1]

    def evaluate_entry(self, df: pd.DataFrame) -> bool:
        if len(df) < max(self.rsi_period, self.vol_window, self.atr_period, self.ma_long):
            return False

        current_time = df.index[-1]
        
        if self.current_consecutive_losses >= self.max_consecutive_losses:
            return False
            
        if not self.can_trade_today(current_time):
            return False

        self.adjust_parameters(df)
        
        # BTC 전용 VWAP 필터
        if self.is_btc:
            vwap = self.calculate_vwap(df)
            price_diff = abs(df['close'].iloc[-1] / vwap - 1)
            if price_diff > self.vwap_threshold:
                return False
        
        if not (self.is_high_volatility(df) and self.is_high_volume(df)):
            return False

        trend_strength = self.calculate_trend_strength(df)
        if trend_strength > self.trend_threshold:
            return False

        ma_short = df['close'].rolling(window=self.ma_short).mean()
        ma_mid = df['close'].rolling(window=self.ma_mid).mean()
        
        if self.is_btc:
            # BTC는 단기/중기 이평선 수렴 구간 진입
            ma_diff_ratio = abs(ma_short.iloc[-1] / ma_mid.iloc[-1] - 1)
            if ma_diff_ratio > 0.001:  # 0.1% 이상 차이나면 거래 안함
                return False
        elif self.is_doge:
            # DOGE는 단기 이평선이 중기 이평선 근처에서 움직일 때
            ma_diff_ratio = abs(ma_short.iloc[-1] / ma_mid.iloc[-1] - 1)
            if ma_diff_ratio > 0.005:  # 0.5% 이상 차이나면 거래 안함
                return False
        else:
            if ma_short.iloc[-1] > ma_mid.iloc[-1]:
                return False

        rsi = self.calculate_rsi(df)
        rsi_values = rsi.iloc[-3:]
        
        if self.is_btc:
            # BTC는 RSI가 하락 후 횡보하는 구간 포착
            rsi_std = rsi_values.std()
            if not (rsi_values.iloc[-1] < self.entry_threshold and
                   rsi_std < 1.0):  # RSI 변동성이 작을 때
                return False
        elif self.is_doge:
            # DOGE는 RSI가 일정 수준 이하이고 상승 반전 시
            if not (rsi_values.iloc[-1] < self.entry_threshold and
                   rsi_values.iloc[-1] > rsi_values.iloc[-2]):
                return False
        else:
            if not (rsi_values.iloc[-1] < self.entry_threshold and 
                   rsi_values.is_monotonic_decreasing):
                return False

        return True

    def evaluate_exit(self, df: pd.DataFrame, entry_price: float, hold_time: float) -> Tuple[bool, str]:
        if len(df) < self.rsi_period or hold_time < self.min_hold_time:
            return False, ""

        # 파라미터 동적 조정
        self.adjust_parameters(df)
        
        current_price = df.iloc[-1]['close']
        profit_ratio = (current_price / entry_price - 1)
        
        # 추세 반전 확인
        trend_strength = self.calculate_trend_strength(df)
        if trend_strength > self.trend_threshold and profit_ratio > 0:
            return True, "추세 반전 익절"

        if profit_ratio <= self.stop_loss:
            return True, "손절"
        if profit_ratio >= self.take_profit:
            return True, "익절"
        if profit_ratio > self.take_profit * 0.5:
            high_price = df['high'].iloc[-int(hold_time):].max()
            if (current_price / high_price - 1) <= -self.trailing_stop:
                return True, "트레일링 스탑"
        
        rsi = self.calculate_rsi(df)
        if rsi.iloc[-1] > self.exit_threshold:
            return True, "RSI 반등 종료"
        if hold_time >= self.max_hold_time:
            return True, "시간초과"
            
        return False, ""

def fetch_ohlcv(ticker: str, start_date: str, end_date: str, interval: str = "minute5") -> pd.DataFrame:
    """OHLCV 데이터를 안전하게 가져옵니다."""
    if not check_ticker_validity(ticker):
        logging.error(f"{ticker}: 유효하지 않은 티커")
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    all_df = pd.DataFrame()
    to = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    retry_count = 0
    max_retries = 5  # 재시도 횟수 증가

    while True:
        try:
            if retry_count >= max_retries:
                logging.error(f"{ticker}: 최대 재시도 횟수 초과")
                break
                
            df = safe_request_with_retry(
                pyupbit.get_ohlcv,
                ticker,
                interval=interval,
                count=200,
                to=to
            )
            
            if df is None or df.empty:
                logging.warning(f"{ticker}: 데이터 수신 실패 (to: {to})")
                retry_count += 1
                time.sleep(2)  # 대기 시간 증가
                continue
                
            all_df = pd.concat([df, all_df])
            all_df = all_df[~all_df.index.duplicated(keep='first')]
            
            to = df.index[0] - timedelta(minutes=5)
            
            if all_df.index[0].date() <= start_dt.date():
                break
                
            time.sleep(0.5)  # API 부하 방지
            retry_count = 0
            
        except Exception as e:
            logging.error(f"데이터 수집 중 오류 발생 ({ticker}): {str(e)}")
            retry_count += 1
            time.sleep(2)
            continue

    if all_df.empty:
        logging.warning(f"{ticker}: 수집된 데이터가 없습니다")
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    filtered_df = all_df[all_df.index.date >= start_dt.date()]
    
    if filtered_df.empty:
        logging.warning(f"{ticker}: 필터링 후 데이터가 없습니다")
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
    if not filtered_df.index.is_monotonic_increasing:
        filtered_df = filtered_df.sort_index()
    
    logging.info(f"{ticker}: {len(filtered_df)}개 데이터 수집 완료")
    return filtered_df

def backtest_strategy(strategy, df):
    """백테스트 실행 (성능 최적화 버전)"""
    # 미리 계산할 수 있는 지표들을 한번에 계산
    rsi = strategy.calculate_rsi(df)
    atr = strategy.calculate_atr(df)
    volatility = df['high'] / df['low'] - 1
    vol_ma = volatility.rolling(window=strategy.vol_window).mean()
    vol_threshold = volatility.rolling(window=strategy.vol_window).quantile(strategy.vol_percentile/100)
    
    trades = []
    holding = False
    entry_price = None
    entry_time = None
    
    # 벡터화된 계산을 위한 준비
    prices = df['close'].values
    highs = df['high'].values
    times = df.index.values
    
    for i in range(max(strategy.rsi_period, strategy.vol_window, strategy.atr_period), len(df)):
        if len(trades) >= 1000:  # 최대 거래 수 제한
            break
            
        current_price = prices[i]
        current_time = times[i]
        
        # 변동성 체크 (미리 계산된 값 사용)
        atr_ratio = atr.iloc[i] / current_price
        is_volatile = (atr_ratio > strategy.min_volatility and 
                      vol_ma.iloc[i] > vol_threshold.iloc[i])
        
        if not holding:
            # 진입 조건 체크 (미리 계산된 RSI 사용)
            if is_volatile and rsi.iloc[i] < strategy.entry_threshold:
                holding = True
                entry_price = current_price
                entry_time = current_time
        
        elif holding:
            # 청산 조건 체크
            hold_time = (pd.Timestamp(current_time) - pd.Timestamp(entry_time)).total_seconds() / 60
            profit_ratio = (current_price / entry_price - 1)
            
            exit_flag = False
            reason = ""
            
            # 청산 조건 체크 (단순화된 버전)
            if hold_time < strategy.min_hold_time:
                continue
                
            if profit_ratio <= strategy.stop_loss:
                exit_flag, reason = True, "손절"
            elif profit_ratio >= strategy.take_profit:
                exit_flag, reason = True, "익절"
            elif not is_volatile and profit_ratio > 0:
                exit_flag, reason = True, "변동성 감소 익절"
            elif not is_volatile and hold_time > strategy.min_hold_time * 2:
                exit_flag, reason = True, "변동성 감소 청산"
            elif rsi.iloc[i] > strategy.exit_threshold:
                exit_flag, reason = True, "RSI 반등 종료"
            elif hold_time >= strategy.max_hold_time:
                exit_flag, reason = True, "시간초과"
            
            if exit_flag:
                trades.append({
                    "entry": entry_time,
                    "exit": current_time,
                    "pnl": profit_ratio * 100,
                    "reason": reason
                })
                holding = False

    return pd.DataFrame(trades)

def analyze_results(df_trades: pd.DataFrame) -> Dict:
    if df_trades.empty:
        return {
            "trades": 0,
            "win_rate": 0,
            "total_return": 0,
            "sharpe": 0
        }
    wins = df_trades[df_trades['pnl'] > 0]
    pnl_std = df_trades['pnl'].std()
    return {
        "trades": len(df_trades),
        "win_rate": len(wins) / len(df_trades) * 100,
        "total_return": df_trades['pnl'].sum(),
        "sharpe": np.sqrt(252) * df_trades['pnl'].mean() / pnl_std if pnl_std != 0 else 0
    }

def print_trade_summary(ticker: str, result: Dict) -> None:
    """거래 결과 요약을 예쁘게 출력"""
    print("\n" + "="*50)
    print(f"🪙 {ticker} 백테스트 결과")
    print("-"*50)
    print(f"📊 총 거래 횟수: {result['trades']}회")
    print(f"✨ 승률: {result['win_rate']:.2f}%")
    
    # 수익률에 따른 이모지 선택
    if result['total_return'] > 0:
        return_emoji = "🔥"
    elif result['total_return'] < 0:
        return_emoji = "📉"
    else:
        return_emoji = "➖"
    print(f"{return_emoji} 총 수익률: {result['total_return']:.2f}%")
    
    # 샤프비율에 따른 이모지 선택
    if result['sharpe'] > 1:
        sharpe_emoji = "⭐"
    elif result['sharpe'] > 0:
        sharpe_emoji = "✅"
    else:
        sharpe_emoji = "⚠️"
    print(f"{sharpe_emoji} 샤프비율: {result['sharpe']:.2f}")
    print("="*50)

def run_multi_coin_backtest(tickers: List[str], start_date: str, end_date: str):
    """멀티 코인 백테스트 실행"""
    summary = {}
    
    print("\n📈 백테스트 시작")
    print(f"📅 기간: {start_date} ~ {end_date}")
    print(f"🎯 대상 코인: {', '.join(tickers)}\n")
    
    total_tickers = len(tickers)
    
    for idx, ticker in enumerate(tickers, 1):
        try:
            print(f"\n⏳ 진행률: {idx}/{total_tickers} - {ticker} 분석 중...")
            
            df = fetch_ohlcv(ticker, start_date, end_date)
            if df.empty:
                logging.warning(f"{ticker}: 데이터 없음, 건너뜀")
                continue
            
            # 코인별로 전략 인스턴스 생성
            strategy = RSIReversalStrategy(ticker)
            trades = backtest_strategy(strategy, df)
            result = analyze_results(trades)
            summary[ticker] = result
            
            if not trades.empty:
                trades.to_csv(f"logs/trades_{ticker}_{start_date}_{end_date}.csv")
            
            print_trade_summary(ticker, result)
            
        except Exception as e:
            logging.error(f"{ticker} 백테스트 중 오류 발생: {str(e)}")
            summary[ticker] = {
                "trades": 0,
                "win_rate": 0,
                "total_return": 0,
                "sharpe": 0
            }
    
    result_df = pd.DataFrame(summary).T
    
    print("\n🎉 전체 백테스트 완료!")
    print("\n=== 📊 전략 성과 요약 ===")
    print(f"📈 평균 승률: {result_df['win_rate'].mean():.2f}%")
    print(f"💰 평균 수익률: {result_df['total_return'].mean():.2f}%")
    print(f"📊 평균 샤프비율: {result_df['sharpe'].mean():.2f}")
    print(f"🔄 총 거래횟수: {result_df['trades'].sum():.0f}회")
    
    # 최고/최저 성과 코인 표시
    best_coin = result_df['total_return'].idxmax()
    worst_coin = result_df['total_return'].idxmin()
    print(f"\n🏆 최고 성과: {best_coin} ({result_df.loc[best_coin, 'total_return']:.2f}%)")
    print(f"⚠️ 최저 성과: {worst_coin} ({result_df.loc[worst_coin, 'total_return']:.2f}%)")
    
    return result_df

if __name__ == "__main__":
    # BTC를 첫번째로, DOGE를 두번째로 순서 변경
    altcoins = ["KRW-BTC", "KRW-DOGE", "KRW-XRP", "KRW-SOL", "KRW-SAND", "KRW-ARB"]
    start = "2025-01-1"
    end = "2025-04-15"

    # 결과 저장 디렉토리 생성
    os.makedirs("logs", exist_ok=True)

    try:
        result_df = run_multi_coin_backtest(altcoins, start, end)
        result_df.to_csv("logs/rsi_reversal_multi_result.csv")
        print("\n전체 결과:\n", result_df)
        
    except Exception as e:
        logging.error(f"프로그램 실행 중 오류 발생: {str(e)}")
