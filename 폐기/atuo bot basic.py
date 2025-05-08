#!/usr/bin/env python3
import os
import time
import pyupbit
import logging
from dotenv import load_dotenv
from datetime import datetime
from colorama import init, Fore, Style

# 터미널 색상 초기화 (자동 리셋)
init(autoreset=True)

# ★ 커스텀 로깅 포매터 (컬러 적용 및 간결한 출력)
class CustomFormatter(logging.Formatter):
    blue_format = Style.BRIGHT + Fore.BLUE + "%(asctime)s [%(levelname)s] " + Fore.WHITE + "%(message)s" + Style.RESET_ALL
    green_format = Style.BRIGHT + Fore.GREEN + "%(asctime)s [%(levelname)s] " + Fore.WHITE + "%(message)s" + Style.RESET_ALL
    yellow_format = Style.BRIGHT + Fore.YELLOW + "%(asctime)s [%(levelname)s] " + Fore.WHITE + "%(message)s" + Style.RESET_ALL
    red_format = Style.BRIGHT + Fore.RED + "%(asctime)s [%(levelname)s] " + Fore.WHITE + "%(message)s" + Style.RESET_ALL

    FORMATS = {
        logging.DEBUG: blue_format,
        logging.INFO: green_format,
        logging.WARNING: yellow_format,
        logging.ERROR: red_format,
        logging.CRITICAL: red_format
    }
    
    def format(self, record):
        fmt = self.FORMATS.get(record.levelno, self.green_format)
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

# 콘솔 핸들러와 파일 핸들러에 CustomFormatter 적용
console_handler = logging.StreamHandler()
console_handler.setFormatter(CustomFormatter())
file_handler = logging.FileHandler("trading_log.txt", encoding="utf-8")
file_handler.setFormatter(CustomFormatter())
logging.basicConfig(handlers=[console_handler, file_handler], level=logging.INFO)

# ★ 환경변수 로드 및 API 키 설정
load_dotenv()
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

# ★ 설정값 (기본 전략 및 거래 조건)
SETTINGS = {
    "K": 0.25,                   # 변동성 돌파 전략 계수
    "CHECK_INTERVAL": 60,        # 60초마다 사이클 실행
    "RSI_PERIOD": 14,
    "RSI_OVERSOLD": 30,
    "BB_WINDOW": 20,
    "BB_STD_DEV": 2,
    "STABLE_COINS": ["KRW-USDT", "KRW-DAI", "KRW-TUSD"],
    "INVEST_RATIO": 0.1,         # 투자 비율 (잔고의 10%)
    "MIN_ORDER_AMOUNT": 50000,   # 최소 주문 금액
    "TAKE_PROFIT_PERCENT": 5,    # 익절 조건 (5%)
    "STOP_LOSS_PERCENT": 5       # 손절 조건 (5%)
}

# Upbit API 객체 생성
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

def get_non_stable_coins():
    """
    KRW 마켓의 코인 중 스테이블 코인을 제외한 리스트를 반환합니다.
    """
    tickers = pyupbit.get_tickers(fiat="KRW")
    non_stable = [ticker for ticker in tickers if ticker not in SETTINGS["STABLE_COINS"]]
    logging.info(f"대상 코인 리스트: {non_stable}")
    return non_stable

def calculate_target_price(ticker, k):
    """
    변동성 돌파 전략으로 목표가를 계산합니다.
    (전일 종가 + 전일 변동폭 * k)
    """
    df = pyupbit.get_ohlcv(ticker, interval="day", count=2)
    # DataFrame이 None이거나, 비어있거나, 2개 미만의 데이터이면 부족한 것으로 판단
    if df is None or df.empty or len(df) < 2:
        logging.warning(f"{ticker} OHLCV 데이터 부족")
        return None
    yesterday = df.iloc[-2]
    target = yesterday['close'] + (yesterday['high'] - yesterday['low']) * k
    logging.info(f"{ticker} 목표가: {target:.2f} (어제 종가: {yesterday['close']:.2f}, 변동폭: {yesterday['high'] - yesterday['low']:.2f})")
    return target

def calculate_rsi(prices, period=14):
    """간단한 RSI 계산"""
    if len(prices) < period:
        logging.debug("RSI 계산을 위한 데이터 부족")
        return None
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [delta for delta in deltas if delta > 0]
    losses = [-delta for delta in deltas if delta < 0]
    avg_gain = sum(gains[-period:]) / period if gains else 0
    avg_loss = sum(losses[-period:]) / period if losses else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bb_position(df):
    """볼린저 밴드 위치 계산 (종가 데이터 기반)"""
    # DataFrame이 없거나, 비어있거나, 윈도우 길이 미만이면 계산 불가
    if df is None or df.empty or len(df) < SETTINGS["BB_WINDOW"]:
        logging.warning("BB 포지션 계산을 위한 데이터 부족")
        return None
    rolling_mean = df['close'].rolling(window=SETTINGS["BB_WINDOW"]).mean()
    rolling_std = df['close'].rolling(window=SETTINGS["BB_WINDOW"]).std()
    upper = rolling_mean + (SETTINGS["BB_STD_DEV"] * rolling_std)
    lower = rolling_mean - (SETTINGS["BB_STD_DEV"] * rolling_std)
    current_price = df['close'].iloc[-1]
    position = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
    return position

def calculate_total_score(rsi, bb_pos):
    """
    RSI와 볼린저 밴드 포지션을 기반으로 단순 신뢰도 점수를 계산합니다.
    """
    rsi_score = max(0, 100 - (rsi - SETTINGS["RSI_OVERSOLD"]) * 2) if rsi is not None else 0
    bb_score = max(0, 100 - (bb_pos - 0.2) * 500) if bb_pos is not None else 0
    total = 0.5 * rsi_score + 0.5 * bb_score
    return total

def should_buy(rsi, bb_pos):
    """
    매수 조건 평가:
      - RSI와 BB 포지션을 기반으로 계산한 총점이 70 이상이면 매수 조건 만족
    """
    score = calculate_total_score(rsi, bb_pos)
    logging.info(f"매수 평가 -> RSI: {rsi}, BB 포지션: {bb_pos}, 점수: {score:.2f}")
    return score >= 70

def execute_buy(ticker, current_price):
    """
    매수 주문 실행 (실제 주문은 주석 처리되어 있으며, 대신 로그로 기록합니다.)
    """
    krw_balance = upbit.get_balance("KRW")
    invest_amt = krw_balance * SETTINGS["INVEST_RATIO"]
    if invest_amt < SETTINGS["MIN_ORDER_AMOUNT"]:
        logging.warning(f"{ticker} 매수 불가: 잔고 부족 ({invest_amt:.0f}원)")
        return
    # 실제 주문 호출 (예: upbit.buy_market_order(ticker, invest_amt))는 여기에 추가합니다.
    logging.info(f"{ticker} 매수 실행 -> 투자금액: {invest_amt:.0f}원, 현재가: {current_price:.2f}")

def execute_sell(ticker, buy_price):
    """
    익절/손절 조건에 따라 매도 주문을 실행합니다.
    """
    coin_balance = upbit.get_balance(ticker)
    current_price = pyupbit.get_current_price(ticker)
    if current_price is None:
        logging.warning(f"{ticker} 현재가 조회 실패")
        return
    tp_price = buy_price * (1 + SETTINGS["TAKE_PROFIT_PERCENT"] / 100)
    sl_price = buy_price * (1 - SETTINGS["STOP_LOSS_PERCENT"] / 100)
    if current_price >= tp_price:
        # 예: upbit.sell_market_order(ticker, coin_balance)
        logging.info(f"{ticker} 익절 매도 -> 매수가: {buy_price:.2f}, 현재가: {current_price:.2f}")
    elif current_price <= sl_price:
        # 예: upbit.sell_market_order(ticker, coin_balance)
        logging.info(f"{ticker} 손절 매도 -> 매수가: {buy_price:.2f}, 현재가: {current_price:.2f}")
    else:
        logging.debug(f"{ticker} 매도 조건 미충족 -> 매수가: {buy_price:.2f}, 현재가: {current_price:.2f}")

def main():
    logging.info("자동매매 시작")
    while True:
        non_stable = get_non_stable_coins()
        for ticker in non_stable:
            try:
                target = calculate_target_price(ticker, SETTINGS["K"])
                if target is None:
                    continue
                df = pyupbit.get_ohlcv(ticker, interval="minute5", count=100)
                # DataFrame이 None이거나 비어있거나 데이터 개수가 부족한 부분 수정
                if df is None or df.empty or len(df) < 100:
                    logging.warning(f"{ticker} 5분봉 데이터 부족")
                    continue
                current_price = df['close'].iloc[-1]
                if current_price is None:
                    continue
                rsi = calculate_rsi(df['close'].tolist(), SETTINGS["RSI_PERIOD"])
                bb_pos = calculate_bb_position(df)
                if current_price > target and should_buy(rsi, bb_pos):
                    logging.info(f"{ticker} 매수 조건 만족 -> 현재가: {current_price:.2f}, 목표가: {target:.2f}")
                    execute_buy(ticker, current_price)
                    # 매수 후 익절/손절 모니터링 (예시로 60초 동안 5초 간격으로 체크)
                    monitor_start = time.time()
                    while time.time() - monitor_start < SETTINGS["CHECK_INTERVAL"]:
                        execute_sell(ticker, current_price)
                        time.sleep(5)
            except Exception as e:
                logging.error(f"{ticker} 처리 오류: {e}")
        logging.info("사이클 완료. 다음 사이클 전 대기 중...")
        time.sleep(SETTINGS["CHECK_INTERVAL"])

if __name__ == "__main__":
    main()
