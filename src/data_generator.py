import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_data(start_date: str, end_date: str, output_file: str):
    """가격 데이터 생성"""
    # 시작일과 종료일 설정
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # 1시간 간격의 시간 인덱스 생성
    dates = pd.date_range(start=start, end=end, freq='H')
    
    # 초기 가격 설정
    initial_price = 140000000  # 1.4억원
    
    # 가격 변동성 설정
    volatility = 0.002  # 0.2%
    
    # 가격 데이터 생성
    np.random.seed(42)  # 재현성을 위한 시드 설정
    returns = np.random.normal(0, volatility, len(dates))
    prices = initial_price * (1 + returns).cumprod()
    
    # OHLCV 데이터 생성
    data = []
    for i in range(len(dates)):
        price = prices[i]
        high = price * (1 + abs(np.random.normal(0, volatility)))
        low = price * (1 - abs(np.random.normal(0, volatility)))
        volume = np.random.lognormal(4, 1)  # 거래량
        
        data.append({
            'timestamp': dates[i],
            'open': price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    # DataFrame 생성
    df = pd.DataFrame(data)
    
    # 결과 저장
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"데이터가 생성되었습니다: {output_file}")
    print(f"데이터 크기: {df.shape}")
    print(f"기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

if __name__ == '__main__':
    # 2023년 1월 1일부터 2025년 5월 16일까지의 데이터 생성
    generate_data(
        start_date='2023-01-01',
        end_date='2025-05-16',
        output_file='data/raw/KRW-BTC_1h.csv'
    ) 