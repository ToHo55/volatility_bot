# Python 3.9 베이스 이미지 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 필요한 Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY src/ ./src/
COPY config/ ./config/

# 로그 디렉토리 생성
RUN mkdir -p logs

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV TZ=Asia/Seoul

# 실행 명령
CMD ["python", "src/main.py"] 