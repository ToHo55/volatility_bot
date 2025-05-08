#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
from backtesting.backtest_engine import run_backtest

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('backtest.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='암호화폐 트레이딩 봇 백테스팅')
    
    parser.add_argument('--start_date', type=str, required=True,
                      help='백테스팅 시작일 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True,
                      help='백테스팅 종료일 (YYYY-MM-DD)')
    parser.add_argument('--initial_balance', type=float, default=10000000,
                      help='초기 자본금 (원)')
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging()
    
    # 백테스팅 실행
    run_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_balance=args.initial_balance
    )

if __name__ == '__main__':
    main() 