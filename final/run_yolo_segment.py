import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# 1. 경로 설정 (방금 학습된 train3 폴더의 best.pt를 가져옵니다!)
MODEL_PATH = '/data3/home/h22000561/psa_grading/runs/segment/train3/weights/best.pt' 
INPUT_DIR = '/data3/home/h22000561/psa_grading/data/test'
OUTPUT_DIR = '/data3/home/h22000561/psa_grading/yolo_pure_cards'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(">>> 🧠 학습된 YOLOv8 누끼 마스터 모델을 불러옵니다...")
model = YOLO(MODEL_PATH)

# 이미지 탐색
all_images = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
    all_images.extend(Path(INPUT_DIR).rglob(ext))

print(f">>> 총 {len(all_images)}장의 슬랩(Slab) 이미지에서 순수 카드만 도려냅니다!")

# 2. AI 누끼 추출 루프
for i, img_path in enumerate(all_images, 1):
    img = cv2.imread(str(img_path))
    if img is None: continue

    # YOLO 추론
    results = model.predict(source=img, conf=0.5, verbose=False)
    result = results[0]
    
    if result.masks is None or len(result.masks) == 0:
        print(f"[경고] {img_path.name} - 카드를 찾지 못했습니다.")
        continue

    # 마스크(누끼 영역) 가져오기 및 원본 크기에 맞게 조절
    mask = result.masks.data[0].cpu().numpy()
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask * 255).astype(np.uint8)
    
    # 원본에 마스크 씌워서 배경 날리기
    pure_card = cv2.bitwise_and(img, img, mask=mask)
    
    # 바운딩 박스를 이용해 딱 맞게 자르기
    box = result.boxes.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box
    final_cropped = pure_card[y1:y2, x1:x2]
    
    # 저장
    save_name = f"yolo_{img_path.name}"
    cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), final_cropped)
    
    if i % 20 == 0 or i == len(all_images):
        print(f"  [{i}/{len(all_images)}] 장 일괄 누끼 처리 완료...")

print(f"\n✅ YOLO 기반 무결점 누끼 추출 완료! 저장 위치: {OUTPUT_DIR}")
