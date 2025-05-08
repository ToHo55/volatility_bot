# 개발 일지

## 2024-05-08

### 초기 프로젝트 설정
1. 프로젝트 구조 생성
   - data/ (raw, processed)
   - src/
   - tests/
   - logs/
   - config/
   - docs/

2. 기본 파일 생성
   - requirements.txt: 필요한 패키지 정의
   - README.md: 프로젝트 설명 및 사용법

### 데이터 계층 구현 (Day 1)
1. datasource.py 구현 완료
   - DataSource 클래스 구현
   - 주요 기능:
     * Upbit API 연동 (1분당 60회 호출 제한 준수)
     * CSV 캐시 시스템 구현
     * 히스토리컬 데이터 조회 기능
   - 구현된 메서드:
     * get_upbit_ohlcv(): Upbit에서 OHLCV 데이터 조회
     * save_to_cache(): 데이터를 CSV 캐시로 저장
     * load_from_cache(): 캐시에서 데이터 로드
     * get_historical_data(): 지정된 기간의 히스토리컬 데이터 조회

### 지표 계산 모듈 구현 (½ day)
1. indicators.py 구현 완료
   - TechnicalIndicators 클래스 구현
   - 주요 기능:
     * RSI 계산 (0-100 사이로 클리핑)
     * EMA 계산 (5일, 20일)
     * EMA 기울기 계산
     * ATR 계산
   - 구현된 메서드:
     * calc_rsi(): RSI 계산
     * calc_ema(): EMA 계산
     * ema_slope(): EMA 기울기 계산
     * calc_atr(): ATR 계산
     * add_indicators(): 모든 지표를 한 번에 추가

### 시그널 엔진 구현 (Day 2)
1. signals.py 구현 완료
   - SignalGenerator 클래스 구현
   - 주요 기능:
     * 매매 신호 생성 (롱 포지션)
     * 손절가 계산 및 관리
     * 포지션 관리
   - 구현된 메서드:
     * generate_signals(): 매매 신호 생성
     * _calculate_stop_price(): 손절가 계산
     * _check_stop_loss(): 손절 조건 확인
   - 매매 전략:
     * 진입 조건: RSI 과매도(30) + 골든 크로스 + EMA5 상승
     * 청산 조건: RSI 과매수(55) 또는 EMA5 하락
     * 손절: 고점 대비 -0.8%

### 백테스트 엔진 구현 (Day 2-3)
1. backtester.py 구현 완료
   - Backtester 클래스 구현
   - 주요 기능:
     * 거래 시뮬레이션
     * 성과 지표 계산
     * 결과 시각화
   - 구현된 메서드:
     * run(): 백테스트 실행
     * _simulate_trades(): 거래 시뮬레이션
     * _calculate_metrics(): 성과 지표 계산
     * print_results(): 결과 출력
   - 성과 지표:
     * 총 수익률, CAGR
     * 샤프 비율
     * 최대 낙폭 (MDD)
     * 승률, 평균 수익/손실
     * 거래 횟수 통계

2. WalkForward 클래스 구현
   - 주요 기능:
     * 학습/테스트 기간 분할
     * 롤링 윈도우 테스트
   - 구현된 메서드:
     * run(): Walk-forward 테스트 실행
   - 성과 지표:
     * 평균 수익률
     * 평균 샤프 비율
     * 평균 승률
     * 전체 거래 통계

### 실거래 실행기 구현 (Day 4)
1. executor.py 구현 완료
   - Order 클래스 구현
     * 주문 정보 관리
     * 주문 상태 추적
   - Position 클래스 구현
     * 포지션 정보 관리
     * PnL 계산
   - Executor 클래스 구현
     * 주요 기능:
       - 주문 실행 및 취소
       - 포지션 관리
       - 전략 실행
       - 상태 모니터링
     * 구현된 메서드:
       - place_order(): 주문 실행
       - cancel_order(): 주문 취소
       - update_position(): 포지션 업데이트
       - execute_strategy(): 전략 실행
       - print_status(): 상태 출력
     * TODO:
       - Upbit API 연동 구현
       - 미체결 주문 처리
       - 에러 처리 강화

### 로깅·모니터링 구현 (½ day)
1. logger.py 구현 완료
   - Logger 클래스 구현
   - 주요 기능:
     * 로그 파일 관리
     * 콘솔 출력 설정
     * 상태 모니터링
   - 구현된 메서드:
     * print_startup(): 시작 메시지 출력
     * print_shutdown(): 종료 메시지 출력
     * print_error(): 에러 메시지 출력
     * print_trade(): 거래 정보 출력
     * print_position(): 포지션 정보 출력
     * print_progress(): 진행 상태 출력
   - 로그 설정:
     * 콘솔 출력: INFO 레벨, 컬러 포맷
     * 파일 출력: DEBUG 레벨, 일별 로테이션
     * 보관 기간: 30일
     * 압축 저장: ZIP 형식

### 다음 단계 구현 계획
1. 테스트 & 검증 (Day 5)
   - pytest 테스트 케이스 작성
   - Monte-Carlo 시뮬레이션

2. 배포 준비 (Day 6-7)
   - API 키 암호화
   - Docker 설정
   - CI/CD 파이프라인 구축 

## 테스트 케이스 구현 (Day 4-5)
- `test_datasource.py` 구현
  - Upbit API 연동 테스트
  - 캐시 시스템 테스트
  - 데이터 정규화 테스트
  - 에러 처리 테스트
- `test_indicators.py` 구현
  - RSI 계산 테스트
  - EMA 계산 테스트
  - EMA 기울기 테스트
  - ATR 계산 테스트
  - 에러 처리 테스트
- `test_signals.py` 구현
  - 시그널 생성 테스트
  - 스탑로스 계산 테스트
  - 진입/청산 조건 테스트
  - 에러 처리 테스트
- `test_backtester.py` 구현
  - 백테스트 실행 테스트
  - 거래 시뮬레이션 테스트
  - 성과 지표 계산 테스트
  - 워크포워드 테스트
  - 에러 처리 테스트
- `test_executor.py` 구현
  - 주문 생성/취소 테스트
  - 포지션 관리 테스트
  - 전략 실행 테스트
  - 에러 처리 테스트
- `test_logger.py` 구현
  - 로거 초기화 테스트
  - 로그 파일 관리 테스트
  - 메시지 출력 테스트
  - 로그 로테이션 테스트

## 다음 단계
1. 배포 준비
   - Docker 컨테이너화
   - CI/CD 파이프라인 구축
   - 모니터링 시스템 구축
2. 문서화
   - API 문서 작성
   - 사용자 매뉴얼 작성
   - 배포 가이드 작성 