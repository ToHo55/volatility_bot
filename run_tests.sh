#!/bin/bash

# PYTHONPATH 설정
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 테스트 실행
pytest tests/ --cov=src/ --cov-report=xml

# 테스트 결과 확인
if [ $? -eq 0 ]; then
    echo "테스트가 성공적으로 완료되었습니다."
    exit 0
else
    echo "테스트 실행 중 오류가 발생했습니다."
    exit 1
fi 