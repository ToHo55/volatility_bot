# RSI 역추세 전략 자동 백테스트 (다중 알트코인)
# 주요 구성: 전략 클래스 + 다중 티커 루프 + 성과 비교 분석

import os
import time
import logging
import pyupbit
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
os.makedirs("logs", exist_ok=True)

class RSIReversalStrategy:
    def __init__(self):
        # RSI 설정
        self.rsi_period = 14
        self.entry_threshold = 35      # 30에서 35로 완화
        self.exit_threshold = 60       # 50에서 60으로 완화
        self.rsi_recovery_level = 55   # 50에서 55로 조정

        # 추가 지표 설정
        self.ma_period = 20          # 이동평균선 기간
        self.volume_ma_period = 20   # 거래량 이동평균 기간
        self.min_volume_ratio = 1.1    # 1.2에서 1.1로 완화
        self.trend_window = 3          # 5에서 3으로 완화

        # 손익 설정
        self.stop_loss = -0.015      # 손절 라인 (완화)
        self.take_profit = 0.02      # 익절 라인 (상향)
        self.trailing_stop = 0.005   # 트레일링 스탑 (상향)
        self.partial_tp_ratio = 0.5  # 부분 익절 비율

        # 시간 설정
        self.min_hold_time = 5         # 10에서 5분으로 완화
        self.max_hold_time = 180     # 최대 홀딩 시간 (분)

        # 거래 비용
        self.fees = {
            'buy_fee': 0.0005,       # 매수 수수료
            'sell_fee': 0.0005,      # 매도 수수료
            'slippage': 0.001        # 슬리피지
        }

    def calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """RSI 계산"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def is_uptrend(self, df: pd.DataFrame) -> bool:
        """상승 추세 여부 확인"""
        ma = df['close'].rolling(window=self.ma_period).mean()
        recent_ma = ma.iloc[-self.trend_window:]
        return recent_ma.is_monotonic_increasing

    def has_volume_surge(self, df: pd.DataFrame) -> bool:
        """거래량 급증 여부 확인"""
        volume_ma = df['volume'].rolling(window=self.volume_ma_period).mean()
        current_volume = df['volume'].iloc[-1]
        return current_volume > volume_ma.iloc[-1] * self.min_volume_ratio

    def calculate_profit_ratio(self, current_price: float, entry_price: float) -> float:
        """수수료와 슬리피지를 포함한 실제 수익률 계산"""
        buy_price = entry_price * (1 + self.fees['buy_fee'] + self.fees['slippage'])
        sell_price = current_price * (1 - self.fees['sell_fee'] - self.fees['slippage'])
        return (sell_price / buy_price) - 1

    def evaluate_entry(self, df: pd.DataFrame) -> bool:
        """진입 조건 평가"""
        if len(df) < max(self.rsi_period, self.ma_period, self.volume_ma_period) + 2:
            return False
            
        rsi = self.calculate_rsi(df)
        
        # 기본 RSI 조건
        last_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        
        last_price = df.iloc[-1]['close']
        prev_price = df.iloc[-2]['close']
        
        # 개선된 진입 조건
        rsi_condition = (
            prev_rsi < self.entry_threshold and  # 과매도
            last_rsi > prev_rsi and             # RSI 상승 반전
            last_price > prev_price             # 가격 상승
        )
        
        # 추가 필터
        trend_condition = self.is_uptrend(df)           # 상승 추세
        volume_condition = self.has_volume_surge(df)    # 거래량 급증
        
        # 모든 조건 충족 필요
        return rsi_condition and trend_condition and volume_condition

    def evaluate_exit(self, df: pd.DataFrame, entry_price: float, hold_time: float) -> Tuple[bool, str]:
        """청산 조건 평가"""
        if len(df) < self.rsi_period or hold_time < self.min_hold_time:
            return False, ""

        current_price = df.iloc[-1]['close']
        profit_ratio = self.calculate_profit_ratio(current_price, entry_price)
        rsi = self.calculate_rsi(df)
        current_rsi = rsi.iloc[-1]

        # 1. 손절 (추세 반전 확인)
        if profit_ratio <= self.stop_loss and not self.is_uptrend(df):
            return True, "손절 (추세 반전)"

        # 2. 익절 (RSI 과매수 확인)
        if profit_ratio >= self.take_profit and current_rsi > 70:
            return True, "익절 (RSI 과매수)"

        # 3. RSI 회복 청산 (거래량 감소 확인)
        if current_rsi >= self.rsi_recovery_level and not self.has_volume_surge(df):
            return True, "RSI 회복 (거래량 감소)"

        # 4. 개선된 트레일링 스탑
        if profit_ratio > self.take_profit * self.partial_tp_ratio:
            # 최근 고점 대비 하락폭 계산 (변동성 고려)
            lookback = min(int(hold_time/2), 20)  # 최대 20개 봉 확인
            high_price = df['high'].iloc[-lookback:].max()
            atr = df['high'].iloc[-lookback:].max() - df['low'].iloc[-lookback:].min()
            trailing_stop = max(self.trailing_stop, atr / high_price)  # 변동성 기반 트레일링 스탑
            
            if (current_price / high_price - 1) <= -trailing_stop:
                return True, "트레일링 스탑"

        # 5. 시간 초과 (추세 및 수익 확인)
        if hold_time >= self.max_hold_time:
            if profit_ratio > 0 and self.is_uptrend(df):
                return True, "시간초과 익절 (추세 유지)"
            else:
                return True, "시간초과 손절 (추세 약화)"

        return False, ""

def fetch_ohlcv(ticker: str, start_date: str, end_date: str, interval: str = "minute5") -> pd.DataFrame:
    """데이터 수집 함수
    
    Args:
        ticker: 코인 티커 (예: KRW-BTC)
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        interval: 봉 간격 (minute1, minute3, minute5, ...)
        
    Returns:
        pd.DataFrame: OHLCV 데이터프레임
    """
    all_df = pd.DataFrame()
    to = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    retry_count = 0
    max_retries = 3

    while True:
        try:
            if retry_count >= max_retries:
                logging.error(f"{ticker}: 최대 재시도 횟수 초과")
                break
                
            df = pyupbit.get_ohlcv(ticker, interval=interval, count=200, to=to)
            
            if df is None or df.empty:
                logging.warning(f"{ticker}: 데이터 수신 실패 (to: {to})")
                retry_count += 1
                time.sleep(1)  # 재시도 전 대기
                continue
                
            all_df = pd.concat([df, all_df])
            
            # 중복 제거
            all_df = all_df[~all_df.index.duplicated(keep='first')]
            
            to = df.index[0] - timedelta(minutes=5)
            
            if all_df.index[0].date() <= start_dt.date():
                break
                
            time.sleep(0.2)  # API 호출 간격
            retry_count = 0  # 성공시 재시도 카운트 초기화
            
        except Exception as e:
            logging.error(f"데이터 수집 중 오류 발생 ({ticker}): {str(e)}")
            retry_count += 1
            time.sleep(1)
            continue

    if all_df.empty:
        logging.warning(f"{ticker}: 수집된 데이터가 없습니다")
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    # 기간 필터링
    filtered_df = all_df[all_df.index.date >= start_dt.date()]
    
    if filtered_df.empty:
        logging.warning(f"{ticker}: 필터링 후 데이터가 없습니다")
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
    # 데이터 정합성 체크
    if not filtered_df.index.is_monotonic_increasing:
        filtered_df = filtered_df.sort_index()
    
    logging.info(f"{ticker}: {len(filtered_df)}개 데이터 수집 완료")
    return filtered_df

def backtest_strategy(strategy, df):
    """전략 백테스트 실행
    
    Args:
        strategy: 백테스트할 전략 객체
        df: OHLCV 데이터프레임
        
    Returns:
        pd.DataFrame: 거래 기록
    """
    trades = []
    holding = False
    entry_price = None
    entry_time = None
    
    logging.info(f"백테스트 시작: 총 {len(df)} 개의 데이터")
    
    for i in range(len(df)):
        current_df = df.iloc[:i+1]
        if len(current_df) < strategy.rsi_period + 2:
            continue

        now = current_df.index[-1]
        price = current_df.iloc[-1]['close']

        if not holding:
            if strategy.evaluate_entry(current_df):
                holding = True
                entry_price = price
                entry_time = now
                logging.info(f"진입: {now} | 가격: {price:,.0f}")

        elif holding:
            hold_time = (now - entry_time).total_seconds() / 60
            exit_flag, reason = strategy.evaluate_exit(current_df, entry_price, hold_time)
            
            if exit_flag:
                pnl = strategy.calculate_profit_ratio(price, entry_price) * 100
                trades.append({
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_time": now,
                    "exit_price": price,
                    "pnl": pnl,
                    "reason": reason,
                    "hold_time": hold_time
                })
                logging.info(f"청산: {now} | 가격: {price:,.0f} | 수익률: {pnl:.2f}% | 이유: {reason}")
                holding = False

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        logging.info("\n=== 백테스트 완료 ===")
        logging.info(f"총 거래 수: {len(trades_df)}건")
        logging.info(f"승률: {len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100:.2f}%")
        logging.info(f"평균 수익률: {trades_df['pnl'].mean():.2f}%")
        logging.info(f"평균 보유시간: {trades_df['hold_time'].mean():.1f}분")
    else:
        logging.warning("거래 없음")
    
    return trades_df

def analyze_results(df_trades: pd.DataFrame) -> Dict:
    """백테스트 결과 분석
    
    Args:
        df_trades: 거래 기록 데이터프레임
        
    Returns:
        Dict: 분석 결과
    """
    if df_trades.empty:
        return {
            "trades": 0,
            "win_rate": 0,
            "total_return": 0,
            "avg_return": 0,
            "max_return": 0,
            "min_return": 0,
            "avg_hold_time": 0,
            "sharpe": 0,
            "profit_factor": 0
        }
        
    wins = df_trades[df_trades['pnl'] > 0]
    losses = df_trades[df_trades['pnl'] < 0]
    
    total_profit = wins['pnl'].sum() if len(wins) > 0 else 0
    total_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    pnl_std = df_trades['pnl'].std()
    sharpe = np.sqrt(252) * df_trades['pnl'].mean() / pnl_std if pnl_std != 0 else 0
    
    return {
        "trades": len(df_trades),
        "win_rate": len(wins) / len(df_trades) * 100,
        "total_return": df_trades['pnl'].sum(),
        "avg_return": df_trades['pnl'].mean(),
        "max_return": df_trades['pnl'].max(),
        "min_return": df_trades['pnl'].min(),
        "avg_hold_time": df_trades['hold_time'].mean(),
        "sharpe": sharpe,
        "profit_factor": profit_factor
    }

def run_single_rsi_test(ticker: str, start_date: str, end_date: str, slippage: float, fee: float):
    """단일 RSI 전략 테스트 실행
    
    Args:
        ticker: 코인 티커
        start_date: 시작일
        end_date: 종료일
        slippage: 슬리피지
        fee: 수수료
    """
    strategy = RSIReversalStrategy()
    strategy.fees['slippage'] = slippage
    strategy.fees['buy_fee'] = fee
    strategy.fees['sell_fee'] = fee
    
    logging.info(f"\n{'='*50}")
    logging.info(f"단일 RSI 전략 테스트")
    logging.info(f"티커: {ticker}")
    logging.info(f"기간: {start_date} ~ {end_date}")
    logging.info(f"슬리피지: {slippage:.4f}")
    logging.info(f"수수료: {fee:.4f}")
    logging.info(f"{'='*50}\n")
    
    df = fetch_ohlcv(ticker, start_date, end_date)
    if df.empty:
        logging.warning(f"{ticker}: 데이터 없음")
        return None
        
    trades = backtest_strategy(strategy, df)
    result = analyze_results(trades)
    
    # 상세 결과 출력
    logging.info("\n=== 전략 성과 분석 ===")
    logging.info(f"총 거래: {result['trades']}회")
    logging.info(f"승률: {result['win_rate']:.2f}%")
    logging.info(f"총 수익률: {result['total_return']:.2f}%")
    logging.info(f"평균 수익률: {result['avg_return']:.2f}%")
    logging.info(f"최대 수익: {result['max_return']:.2f}%")
    logging.info(f"최대 손실: {result['min_return']:.2f}%")
    logging.info(f"평균 보유시간: {result['avg_hold_time']:.1f}분")
    logging.info(f"샤프 비율: {result['sharpe']:.2f}")
    logging.info(f"수익 팩터: {result['profit_factor']:.2f}")
    
    # 결과를 파일에 저장
    result_dict = {
        'ticker': ticker,
        'start_date': start_date,
        'end_date': end_date,
        'slippage': slippage,
        'fee': fee,
        **result
    }
    
    results_df = pd.DataFrame([result_dict])
    results_df.to_csv(f"logs/rsi_test_{ticker}_{start_date}_{end_date}.csv", index=False)
    
    if not trades.empty:
        trades.to_csv(f"logs/trades_{ticker}_{start_date}_{end_date}.csv", index=False)
    
    return result

def run_automated_rsi_experiment():
    MARKET_PHASES = {
        "recent_2025": ("2025-01-01", "2025-04-15")
    }

    ALTCOINS = {
        "major": ["KRW-ETH", "KRW-XRP"],
        "oversold": ["KRW-SAND", "KRW-ARDR"],
        "low_liquidity": ["KRW-CRE", "KRW-LOOM"]
    }

    SLIPPAGE_TEST = [0.001, 0.002, 0.003]
    FEE_TEST = [0.0005, 0.0010, 0.0020]

    all_results = []
    
    for phase_name, (start, end) in MARKET_PHASES.items():
        for group_name, coins in ALTCOINS.items():
            for ticker in coins:
                for slippage in SLIPPAGE_TEST:
                    for fee in FEE_TEST:
                        logging.info(f"[AUTO] {phase_name} | {group_name} | {ticker} | 슬리피지: {slippage} | 수수료: {fee}")
                        try:
                            result = run_single_rsi_test(ticker, start, end, slippage, fee)
                            if result:
                                result_dict = {
                                    'phase': phase_name,
                                    'group': group_name,
                                    'ticker': ticker,
                                    'slippage': slippage,
                                    'fee': fee,
                                    **result
                                }
                                all_results.append(result_dict)
                        except Exception as e:
                            logging.error(f"실패: {ticker} | 오류: {str(e)}")
                            continue

    # 전체 결과를 DataFrame으로 변환하고 저장
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv("logs/automated_rsi_experiment_results.csv", index=False)
        
        # 결과 분석 및 출력
        logging.info("\n=== 자동화 실험 결과 요약 ===")
        
        # 시장 단계별 분석
        for phase in MARKET_PHASES.keys():
            phase_results = results_df[results_df['phase'] == phase]
            if not phase_results.empty:
                logging.info(f"\n[{phase}] 평균 성과:")
                logging.info(f"승률: {phase_results['win_rate'].mean():.2f}%")
                logging.info(f"수익률: {phase_results['total_return'].mean():.2f}%")
                logging.info(f"샤프비율: {phase_results['sharpe'].mean():.2f}")
        
        # 코인 그룹별 분석
        for group in ALTCOINS.keys():
            group_results = results_df[results_df['group'] == group]
            if not group_results.empty:
                logging.info(f"\n[{group}] 평균 성과:")
                logging.info(f"승률: {group_results['win_rate'].mean():.2f}%")
                logging.info(f"수익률: {group_results['total_return'].mean():.2f}%")
                logging.info(f"샤프비율: {group_results['sharpe'].mean():.2f}")

if __name__ == "__main__":
    logging.info("[시작] 자동 RSI 전략 실험 모드 실행")
    run_automated_rsi_experiment()
