import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# --- 경로 설정 ---
RAW_DIR = Path('/data3/home/h22000561/psa_grading/data/raw')
SAVE_DIR = Path('/data3/home/h22000561/psa_grading/data/train_yolo')
MODEL_PATH = '/data3/home/h22000561/psa_grading/runs/segment/train3/weights/best.pt'

print("=======================================================")
print("🔪 YOLO 기반 PSA 카드 정밀 누끼(Crop) 작업을 시작합니다!")
print(f"🧠 로드하는 모델: {MODEL_PATH}")
print("=======================================================")

# --- YOLO 모델 로드 ---
if not os.path.exists(MODEL_PATH):
    print(f"❌ [에러] 모델 파일이 없습니다! 경로를 다시 확인해주세요.")
    exit(1)

model = YOLO(MODEL_PATH)

def yolo_smart_crop(image_path, save_path):
    try:
        img = Image.open(image_path).convert('RGB')
        # 모델 추론 (conf=0.6 이상 확신할 때만 자름)
        results = model.predict(img, conf=0.6, verbose=False)
        
        if len(results[0].boxes) > 0:
            # 가장 높은 확률의 바운딩 박스 추출
            box = results[0].boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            
            # 카드 영역 정밀 크롭
            cropped_img = img.crop((x1, y1, x2, y2))
            cropped_img.save(save_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"  [오류 발생] {image_path.name}: {e}")
        return False

if __name__ == '__main__':
    total_success = 0
    total_fail = 0

    # RAW_DIR 안의 폴더들 중 'psa'라는 이름이 포함된 폴더만 순회
    for grade_dir in RAW_DIR.iterdir():
        if grade_dir.is_dir() and 'psa' in grade_dir.name.lower():
            grade_name = grade_dir.name
            
            # 타겟 폴더 생성 (예: train_yolo/PSA10)
            target_grade_dir = SAVE_DIR / grade_name
            target_grade_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n>>> [{grade_name}] 폴더 변환 시작...")
            
            success_cnt = 0
            fail_cnt = 0
            
            # 해당 폴더 안의 모든 jpg 이미지 처리
            img_list = list(grade_dir.glob('*.jpg'))
            
            for idx, img_path in enumerate(img_list, 1):
                save_path = target_grade_dir / img_path.name
                
                # 이미 변환된 파일이 있으면 패스 (이어하기 기능)
                if save_path.exists():
                    success_cnt += 1
                    continue
                    
                if yolo_smart_crop(str(img_path), str(save_path)):
                    success_cnt += 1
                else:
                    print(f"  - ⚠️ 탐지 실패: {img_path.name}")
                    fail_cnt += 1
                    
                if idx % 100 == 0:
                    print(f"  진행 상황: {idx}/{len(img_list)} 장 처리 완료...")
            
            print(f"✅ [{grade_name}] 완료! 성공: {success_cnt}장 / 실패: {fail_cnt}장")
            total_success += success_cnt
            total_fail += fail_cnt

    print("\n=======================================================")
    print(f"🎉 전체 누끼 작업 완벽 종료!")
    print(f"  - 총 잘라낸 카드: {total_success}장")
    print(f"  - 탐지 실패(수동 확인 필요): {total_fail}장")
    print(f"  - 저장 경로: {SAVE_DIR}")
    print("=======================================================")
