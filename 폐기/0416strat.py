#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import logging
import pyupbit
from datetime import datetime, timedelta
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

# 환경 설정
load_dotenv()
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

KST = ZoneInfo("Asia/Seoul")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 전략 설정
VOLUME_RATIO_THRESHOLD = 3.0       # 거래량 3배 이상
PRICE_INCREASE_THRESHOLD = 0.02    # 2% 이상 상승
TAKE_PROFIT = 0.03                 # 익절 3.0%
STOP_LOSS = 0.015                  # 손절 1.5%
FORCE_SELL_MIN = 10                # 10분 경과 시 강제 매도
BASE_ORDER_KRW = 50000             # 주문 금액
CHECK_INTERVAL = 60                # 1분마다 실행
MIN_7D_AVG_VOLUME = 500000000      # 7일 평균 거래대금 최소 기준 (500백만원)
MAX_REBUY_RATIO = 0.97             # 현재가가 평균단가보다 3% 이상 쌀 경우에만 재매수 허용

# 리스크 관리 설정
MAX_DAILY_LOSS = -0.05  # 하루 -5% 손실 제한
MAX_LOSSES_PER_TICKER = 2  # 동일 종목 손절 최대 허용 횟수
MIN_MARKET_RSI = 40  # 시장 평균 RSI가 이 값 이하일 경우 매수 중지

daily_realized_profit = 0.0
last_reset_day = datetime.now(KST).date()
ticker_loss_count = {}  # 예: {"KRW-XRP": 2}
market_rsi = 50.0  # 기본값

# 보유 종목 기록: {ticker: {'buy_price': float, 'buy_time': datetime}}
holdings = {}

# 수익률 계산 (슬리피지 + 수수료 고려)
def compute_profit_ratio(buy_price, sell_price):
    slip_fee = 0.0015  # 0.15% x 2 (매수 + 매도)
    return ((sell_price * (1 - slip_fee)) / (buy_price * (1 + slip_fee))) - 1

# 일일 손실 한도 체크

def check_daily_loss_limit(profit_percent):
    global daily_realized_profit, last_reset_day
    today = datetime.now(KST).date()
    if today != last_reset_day:
        daily_realized_profit = 0.0
        last_reset_day = today

    daily_realized_profit += profit_percent / 100
    if daily_realized_profit <= MAX_DAILY_LOSS:
        logging.critical(f"🚨 일일 손실 한도 초과: {daily_realized_profit*100:.2f}%")
        return True
    return False

# 시장 상태 업데이트 (전체 RSI 평균)
def update_market_rsi():
    global market_rsi
    try:
        tickers = pyupbit.get_tickers(fiat="KRW")[:30]  # 상위 30개만 분석
        rsi_list = []
        for t in tickers:
            df = pyupbit.get_ohlcv(t, interval="day", count=15)
            if df is not None and len(df) >= 15:
                delta = df['close'].diff().dropna()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(14).mean().iloc[-1]
                avg_loss = loss.rolling(14).mean().iloc[-1]
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                rsi_list.append(rsi)
        if rsi_list:
            market_rsi = sum(rsi_list) / len(rsi_list)
            logging.info(f"[시장 RSI] 평균 RSI: {market_rsi:.2f}")
    except Exception as e:
        logging.warning(f"[시장 RSI 업데이트 실패] {e}")

# 유틸 함수

def get_filtered_tickers():
    candidates = []
    tickers = pyupbit.get_tickers(fiat="KRW")
    for ticker in tickers:
        try:
            df_day = pyupbit.get_ohlcv(ticker, interval="day", count=7)
            if df_day is None or len(df_day) < 7:
                continue
            avg_value = (df_day['close'] * df_day['volume']).mean()
            if avg_value < MIN_7D_AVG_VOLUME:
                continue

            df = pyupbit.get_ohlcv(ticker, interval="minute1", count=4)
            if df is None or len(df) < 4:
                continue

            vol_now = df['volume'].iloc[-1]
            vol_mean = df['volume'].iloc[-4:-1].mean()
            if vol_mean == 0:
                continue
            vol_ratio = vol_now / vol_mean

            price_now = df['close'].iloc[-1]
            price_prev = df['close'].iloc[-4]
            price_change = (price_now - price_prev) / price_prev

            if vol_ratio >= VOLUME_RATIO_THRESHOLD and price_change >= PRICE_INCREASE_THRESHOLD:
                candidates.append((ticker, price_now))
        except Exception as e:
            logging.warning(f"[필터링 오류] {ticker} - {e}")
            continue
    return candidates

def execute_buy(ticker, current_price):
    """매수 실행 함수"""
    try:
        # 이미 보유중인 경우 스킵
        if ticker in holdings:
            return

        # 손절 횟수 체크
        if ticker_loss_count.get(ticker, 0) >= MAX_LOSSES_PER_TICKER:
            logging.info(f"[매수 제한] {ticker}: 손절 횟수 초과")
            return

        # 주문 가능한 KRW 잔고 확인
        balance = upbit.get_balance("KRW")
        if balance < BASE_ORDER_KRW:
            logging.info(f"[잔고 부족] 주문 필요금액: {BASE_ORDER_KRW:,}원, 현재잔고: {balance:,}원")
            return

        # 시장가 매수
        order = upbit.buy_market_order(ticker, BASE_ORDER_KRW)
        if order and 'error' not in order:
            holdings[ticker] = {
                'buy_price': current_price,
                'buy_time': datetime.now(KST)
            }
            logging.info(f"[매수 성공] {ticker}: {current_price:,}원")
        else:
            logging.error(f"[매수 실패] {ticker}: {order}")

    except Exception as e:
        logging.error(f"[매수 오류] {ticker}: {str(e)}")

def monitor_holdings():
    """보유 종목 모니터링 함수"""
    try:
        for ticker in list(holdings.keys()):
            current_price = pyupbit.get_current_price(ticker)
            if not current_price:
                continue

            entry = holdings[ticker]
            buy_price = entry['buy_price']
            buy_time = entry['buy_time']
            hold_time = datetime.now(KST) - buy_time
            profit_ratio = compute_profit_ratio(buy_price, current_price)

            # 익절, 손절, 강제 매도 조건 체크
            should_sell = False
            sell_reason = ""

            if profit_ratio >= TAKE_PROFIT:
                should_sell = True
                sell_reason = "익절"
            elif profit_ratio <= -STOP_LOSS:
                should_sell = True
                sell_reason = "손절"
                ticker_loss_count[ticker] = ticker_loss_count.get(ticker, 0) + 1
            elif hold_time.total_seconds() >= FORCE_SELL_MIN * 60:
                should_sell = True
                sell_reason = "시간 초과"

            if should_sell:
                # 시장가 매도
                volume = upbit.get_balance(ticker)
                if volume > 0:
                    order = upbit.sell_market_order(ticker, volume)
                    if order and 'error' not in order:
                        profit_percent = profit_ratio * 100
                        logging.info(f"[매도 성공] {ticker} ({sell_reason}) - 수익률: {profit_percent:.2f}%")
                        
                        # 일일 손실 한도 체크
                        if check_daily_loss_limit(profit_percent):
                            logging.critical("일일 손실 한도 도달! 프로그램을 종료합니다.")
                            os._exit(1)
                        
                        del holdings[ticker]
                    else:
                        logging.error(f"[매도 실패] {ticker}: {order}")

    except Exception as e:
        logging.error(f"[모니터링 오류] {str(e)}")

# 메인 루프
if __name__ == "__main__":
    logging.info("🔍 스캐너 봇 시작")
    while True:
        try:
            update_market_rsi()
            if market_rsi < MIN_MARKET_RSI:
                logging.warning(f"[시장 관망] 평균 RSI가 낮음 ({market_rsi:.2f} < {MIN_MARKET_RSI})")
                time.sleep(CHECK_INTERVAL)
                continue
            logging.info("스캔 중...")
            new_tickers = get_filtered_tickers()
            for ticker, price in new_tickers:
                execute_buy(ticker, price)
            monitor_holdings()
        except Exception as e:
            logging.exception(f"[메인 루프 오류] {e}")
        time.sleep(CHECK_INTERVAL)