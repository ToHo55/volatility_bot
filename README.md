# Volatility Trading Bot

암호화폐 변동성 트레이딩 봇

## 설치 방법

```bash
pip install -r requirements.txt
```

## 프로젝트 구조

```
.
├── data/
│   ├── raw/          # 원시 데이터
│   └── processed/    # 처리된 데이터
├── src/
│   ├── datasource.py     # 데이터 수집
│   ├── indicators.py     # 기술적 지표
│   ├── signals.py        # 매매 신호
│   ├── backtester.py     # 백테스팅
│   ├── executor.py       # 주문 실행
│   └── main.py          # 메인 실행
├── tests/            # 테스트 코드
├── logs/            # 로그 파일
└── config/          # 설정 파일
```

## 사용 방법

1. 환경 변수 설정
```bash
cp .env.example .env
# .env 파일에 API 키 등 설정
```

2. 백테스팅 실행
```bash
python src/main.py --mode backtest
```

3. 실거래 실행
```bash
python src/main.py --mode live
``` "# volatility_bot" 
