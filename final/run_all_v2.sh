#!/bin/bash

# 실행할 기존 모델 파일 목록
MODELS=("v17" "v18" "v19" "v20" "v_ultimate_pro")

echo "=========================================================="
echo "🔥 2-Stage YOLO 파이프라인: 모델 연속 학습 및 평가 시작 🔥"
echo "=========================================================="

for MODEL in "${MODELS[@]}"; do
    # 1. 새 파일 이름 생성 (예: v17.py -> v17_2.py)
    NEW_FILE="${MODEL}_2.py"
    
    # 2. 원본 파일 복사
    cp "${MODEL}.py" "$NEW_FILE"
    echo ">>> [준비] $NEW_FILE 생성 및 경로 자동 수정 중..."
    
    # 3. 데이터 경로를 _yolo 버전으로 자동 교체 (sed 마법)
    sed -i "s|/data/raw|/data/train_yolo|g" "$NEW_FILE"
    sed -i "s|/data/test|/data/test_yolo|g" "$NEW_FILE"
    
    # 4. 결과 저장 폴더 이름에 _2 붙이기 (예: v17 -> v17_2)
    # 정규식을 이용해 RES_DIR 경로의 마지막 폴더 이름 뒤에 _2를 붙입니다.
    sed -i "s|/v17|/v17_2|g" "$NEW_FILE"
    sed -i "s|/v18|/v18_2|g" "$NEW_FILE"
    sed -i "s|/v19|/v19_2|g" "$NEW_FILE"
    sed -i "s|/v20|/v20_2|g" "$NEW_FILE"
    sed -i "s|/v_ultimate_pro|/v_ultimate_pro_2|g" "$NEW_FILE"
    
    echo ">>> [실행] $NEW_FILE 5-Fold 학습 및 테스트 시작..."
    
    # 5. 모델 실행 (순차적으로 실행됨)
    /home/sonic/anaconda3/bin/python "$NEW_FILE"
    
    echo ">>> [완료] $NEW_FILE 처리가 끝났습니다. 다음 모델로 넘어갑니다."
    echo "----------------------------------------------------------"
done

echo "🎉 모든 YOLO 누끼 적용 모델(v17_2 ~ v_ultimate_pro_2)의 학습과 평가가 완료되었습니다!"
