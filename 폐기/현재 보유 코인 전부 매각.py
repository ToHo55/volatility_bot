# 필수 시스템 라이브러리
import os
import sys
from datetime import datetime, timedelta
from urllib.parse import urlencode
import json
from pathlib import Path
import traceback

# 암호화 및 인증 관련
import hmac
import hashlib
import jwt
import uuid

# 데이터 처리 및 분석
import numpy as np
import pandas as pd

# 네트워크 및 비동기 처리
import requests
from concurrent.futures import ThreadPoolExecutor

# 유틸리티
import logging
import time
from dotenv import load_dotenv

# .env 파일 경로를 명시적으로 지정
load_dotenv(dotenv_path='c:/Users/fre17/Desktop/코딩/변동성 돌파 모델/.env')

# 환경 변수를 직접 설정하여 테스트
os.environ['ACCESS_KEY'] = 'zKPPajNlcZUD16N28DUAf9SEbAXRwUC7DzsqNEnd'
os.environ['SECRET_KEY'] = 'EmAVXdLJhJLPc7iDq0Sy8ttcAI02oqZZxGT0Df23'

# 환경 변수 로드 확인
access_key = os.getenv('ACCESS_KEY')
secret_key = os.getenv('SECRET_KEY')

if not access_key or not secret_key:
    logging.error("API 키가 설정되지 않았습니다.")
else:
    logging.info(f"ACCESS_KEY: {access_key[:4]}****")  # 키의 일부만 출력하여 확인

# 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("upbit_trade_log.txt"),
        logging.StreamHandler()
    ]
)

# .env 파일 로드 및 확인 코드
def check_env_keys():
    """환경 변수 키 확인"""
    access_key = os.getenv('UPBIT_ACCESS_KEY', '')
    secret_key = os.getenv('UPBIT_SECRET_KEY', '')
    
    if not access_key or not secret_key:
        logging.error("API 키가 설정되지 않았습니다.")
        return False
    return True

class UpbitAPI:
    def __init__(self, access_key, secret_key):
        self.base_url = "https://api.upbit.com/v1"
        self.access_key = access_key
        self.secret_key = secret_key

    def create_headers(self, query=None):
        """JWT 헤더 생성"""
        payload = {
            "access_key": self.access_key,
            "nonce": str(uuid.uuid4()),
        }
        
        if query:
            m = hashlib.sha512()
            query_string = urlencode(query).encode()
            m.update(query_string)
            query_hash = m.hexdigest()
            payload["query_hash"] = query_hash
            payload["query_hash_alg"] = "SHA512"

        jwt_token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        
        if isinstance(jwt_token, bytes):
            jwt_token = jwt_token.decode('utf-8')
        
        return {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json"
        }

    def test_api_connection(self):
        """API 연결 테스트"""
        url = f"{self.base_url}/accounts"
        headers = self.create_headers()
        try:
            response = requests.get(url, headers=headers)
            return response.status_code == 200
        except Exception as e:
            logging.error(f"API 테스트 중 오류 발생: {str(e)}")
            return False

    def get_accounts(self):
        """계좌 조회"""
        url = f"{self.base_url}/accounts"
        headers = self.create_headers()
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"계좌 조회 실패: {e}")
            return None

    def place_order(self, market, side, volume=None, price=None, ord_type="market"):
        """주문 요청"""
        params = {
            "market": market,
            "side": side,
            "ord_type": ord_type,
        }
        
        if volume:
            params["volume"] = str(volume)
        if price:
            params["price"] = str(price)
        
        headers = self.create_headers(query=params)
        url = f"{self.base_url}/orders"
        try:
            response = requests.post(url, json=params, headers=headers)
            response_data = response.json()
            
            # 200(OK)와 201(Created) 모두 정상 응답으로 처리
            if response.status_code in [200, 201]:
                logging.info(f"주문 성공 ({market}): {response_data['uuid']}")
                return response_data
            else:
                logging.error(f"주문 실패 ({market}): {response.status_code}")
                logging.error(f"에러 응답: {response_data}")
                return None
            
            time.sleep(1.0)  # 요청 제한 준수
            
        except requests.exceptions.RequestException as e:
            logging.error(f"주문 요청 실패 ({market}): {str(e)}")
            return None
        except ValueError as e:
            logging.error(f"응답 데이터 처리 실패 ({market}): {str(e)}")
            return None

def get_current_price(market):
    """현재가 조회"""
    try:
        url = f"https://api.upbit.com/v1/ticker"
        params = {"markets": market}
        response = requests.get(url, params=params)
        response.raise_for_status()
        result = response.json()
        if result and len(result) > 0:
            return float(result[0]['trade_price'])
        return None
    except Exception as e:
        logging.error(f"{market} 현재가 조회 실패")
        return None

def get_orderbook(market):
    """호가 정보 조회"""
    try:
        url = "https://api.upbit.com/v1/orderbook"
        params = {"markets": market}
        response = requests.get(url, params=params)
        response.raise_for_status()
        result = response.json()
        if result and len(result) > 0:
            return result[0]
        return None
    except Exception as e:
        logging.error(f"{market} 호가 조회 실패: {str(e)}")
        return None

def sell_all_holdings_first_then_handle_small_amounts(upbit):
    """
    1. 먼저 모든 코인 전체 매도 시도
    2. 이후 남은 소액 코인에 대해 추가 매수 후 매도
    """
    logging.info("전체 매도 프로세스 시작")
    
    # 1단계: 전체 매도 시도
    accounts = upbit.get_accounts()
    if not accounts:
        logging.error("계좌 조회 실패")
        return

    # 먼저 모든 코인 매도 시도
    for account in accounts:
        if account['currency'] == 'KRW':
            continue
            
        market = f"KRW-{account['currency']}"
        balance = float(account['balance'])
        
        if balance > 0:
            try:
                logging.info(f"{market}: {balance} 매도 시도 (현재 보유량)")
                
                # 호가 정보 조회
                orderbook = get_orderbook(market)
                if orderbook is None:
                    logging.error(f"{market} 호가 조회 실패, 다음 코인으로 넘어갑니다.")
                    continue
                
                # 매수 1호가로 예상 거래금액 계산
                bid_price = float(orderbook['orderbook_units'][0]['bid_price'])
                expected_value = balance * bid_price
                
                if expected_value < 5000:  # 업비트 최소 거래금액
                    logging.info(f"{market}: 최소 거래금액(5000원) 미만입니다. 예상 거래금액: {expected_value:.0f}원")
                    # 소액인 경우 추가 매수 후 매도 로직으로 이동
                    continue
                
                sell_result = upbit.place_order(
                    market=market,
                    side="ask",
                    volume=str(balance),
                    ord_type="market"
                )
                
                if sell_result:
                    logging.info(f"{market} 매도 요청 완료: {expected_value:.0f}원")
                else:
                    logging.error(f"{market} 매도 실패: API 응답 없음")
                
            except Exception as e:
                logging.error(f"{market} 매도 중 예외 발생: {str(e)}")
                logging.error(traceback.format_exc())
            
            time.sleep(1)
    
    logging.info("전체 매도 완료. 소액 잔량 확인 중...")
    time.sleep(5)
    
    # 2단계: 남은 소액 코인 처리
    new_accounts = upbit.get_accounts()
    for account in new_accounts:
        if account['currency'] == 'KRW':
            continue
            
        market = f"KRW-{account['currency']}"
        balance = float(account['balance'])
        current_price = get_current_price(market)
        
        if not current_price:
            continue
            
        krw_value = balance * current_price
        
        if 0 < krw_value <= 20000:
            logging.info(f"{market}: {krw_value:.0f}원 잔량 발견, 추가 매수 시도")
            
            buy_result = upbit.place_order(
                market=market,
                side="bid",
                price="5000",
                ord_type="price"
            )
            
            if buy_result:
                logging.info(f"{market} 추가 매수 완료, 최종 매도 준비")
                time.sleep(5)
                
                final_accounts = upbit.get_accounts()
                final_balance = None
                for acc in final_accounts:
                    if acc['currency'] == account['currency']:
                        final_balance = float(acc['balance'])
                        break
                
                if final_balance:
                    final_sell_result = upbit.place_order(
                        market=market,
                        side="ask",
                        volume=str(final_balance),
                        ord_type="market"
                    )
                    
                    if final_sell_result:
                        logging.info(f"{market} 최종 매도 완료")
                    else:
                        logging.error(f"{market} 최종 매도 실패")
            else:
                logging.error(f"{market} 추가 매수 실패")
            
            time.sleep(1)

    logging.info("모든 프로세스 완료")

if __name__ == "__main__":
    if check_env_keys():
        upbit_api = UpbitAPI(access_key, secret_key)
        if upbit_api.test_api_connection():
            sell_all_holdings_first_then_handle_small_amounts(upbit_api)
        else:
            logging.error("API 연결 실패")
    else:
        logging.error("환경 변수 설정을 확인하세요")