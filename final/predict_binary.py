import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, resnet18
import torchvision.transforms as T
import sys

# ==========================================
# 1. 이진 분류 구조의 모델 (kaggle_main_binary.py와 동일)
# ==========================================
class GradingBinaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientnet = efficientnet_b4()
        self.efficientnet.classifier = nn.Identity() 
        
        self.corner_extractor = resnet18()
        self.corner_extractor.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1792 + (512 * 4), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 1) # 🔥 1개의 출력 (PSA 10 확률)
        )

    def forward(self, full_img, tl, tr, bl, br):
        f_full = self.efficientnet(full_img)
        f_tl = self.corner_extractor(tl)
        f_tr = self.corner_extractor(tr)
        f_bl = self.corner_extractor(bl)
        f_br = self.corner_extractor(br)
        combined = torch.cat([f_full, f_tl, f_tr, f_bl, f_br], dim=1)
        return self.classifier(combined)

# ==========================================
# 2. 이미지 전처리 및 패치 추출 (평가 모드)
# ==========================================
def process_image(img_path, patch_size=300, manual_crop=False):
    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지를 찾을 수 없습니다: {img_path}")
        sys.exit(1)
        
    ori_h, ori_w = img.shape[:2]
    
    if manual_crop:
        display_img = img.copy()
        win_h = 800
        win_w = int(win_h * (ori_w / ori_h))
        
        cv2.namedWindow('Select Card ROI', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Select Card ROI', win_w, win_h)
        print("마우스로 카드의 영역을 드래그한 후 SPACE나 ENTER를 누르세요.")
        
        roi = cv2.selectROI('Select Card ROI', display_img, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        
        x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        if w == 0 or h == 0:
            print("선택된 영역이 없습니다. 팝업 없이 전체 이미지를 사용합니다.")
            x, y, w, h = 0, 0, ori_w, ori_h
    else:
        # 팝업 없이 전체 이미지를 바로 카드로 간주
        x, y, w, h = 0, 0, ori_w, ori_h
        
    pts = np.array([
        [x, y], [x+w, y], [x+w, y+h], [x, y+h]
    ], dtype="float32")
    
    dst_w, dst_h = 1000, 1400
    dst = np.array([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (dst_w, dst_h))
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    patches = {
        "Full": cv2.resize(warped_rgb, (512, 512)),
        "TL": warped_rgb[0:patch_size, 0:patch_size],
        "TR": warped_rgb[0:patch_size, dst_w-patch_size:dst_w],
        "BL": warped_rgb[dst_h-patch_size:dst_h, 0:patch_size],
        "BR": warped_rgb[dst_h-patch_size:dst_h, dst_w-patch_size:dst_w]
    }
    
    eval_transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return {k: eval_transform(v).unsqueeze(0) for k, v in patches.items()}, warped_rgb

# ==========================================
# 3. 메인 실행 (추론)
# ==========================================
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 🔥 다운로드 받은 이진 분류용 가중치 (정밀도 93.4% 였던 15에폭 추천)
    MODEL_WEIGHTS = os.path.join(SCRIPT_DIR, "psa_binary_epoch_15.pth") 
    
    TEST_IMAGE = os.path.join(SCRIPT_DIR, "test_card.jpg")
    
    print("\n💡 이미지 처리 방식을 선택하세요:")
    print("1 : 마우스로 직접 영역 선택하기 (수동 크롭 - 주변에 다른 배경이 있을 때)")
    print("2 : 이미지 전체 다 쓰기 (자동 처리 - 사진에 카드만 꽉 차게 찍었을 때)")
    choice = input("선택 (1 또는 2) >> ").strip()
    is_manual = True if choice == '1' else False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n사용 장치: {device}")
    
    # 이진 분류 모델 불러오기
    model = GradingBinaryModel().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True))
        model.eval()
        print("✅ 이진 분류 모델(10 vs Not 10) 가중치를 성공적으로 불러왔습니다!")
    except Exception as e:
        print(f"❌ 가중치 로드 실패: {e}\n(파일명이 psa_binary_epoch_15.pth 가 맞는지 확인하세요.)")
        sys.exit(1)
        
    inputs, warped_viz = process_image(TEST_IMAGE, manual_crop=is_manual)
    full = inputs["Full"].to(device)
    tl, tr = inputs["TL"].to(device), inputs["TR"].to(device)
    bl, br = inputs["BL"].to(device), inputs["BR"].to(device)
    
    with torch.no_grad():
         output = model(full, tl, tr, bl, br)
         # Sigmoid 함수를 사용해 출력을 0~1 (0%~100%) 확률로 변환
         prob_10 = torch.sigmoid(output).item() * 100
         prob_not_10 = 100 - prob_10
         
    print("\n" + "="*40)
    print("📈 이진 분류 예측 결과 (PSA 10 검수기)")
    print("="*40)
    print(f"PSA 10일 확률     : {prob_10:.2f}%")
    print(f"Not 10 (8,9) 확률 : {prob_not_10:.2f}%")
        
    # 절반(50%) 이상이면 PSA 10으로 판정
    if prob_10 >= 50.0:
        final_pred = "PSA 10 (Gem Mint)"
        color = (0, 255, 0) # 초록색
    else:
        final_pred = "Not 10 (PSA 8 or 9)"
        color = (0, 0, 255) # 빨간색
        
    print(f"\nAI 판정: ⭐ {final_pred} ⭐")
    
    warped_bgr = cv2.cvtColor(warped_viz, cv2.COLOR_RGB2BGR)
    cv2.putText(warped_bgr, f"Pred: {final_pred} ({max(prob_10, prob_not_10):.1f}%)", 
                (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
    
    # 비율 깨짐 방지용 창 크기 계산
    ori_h, ori_w = warped_bgr.shape[:2]
    win_h = 800
    win_w = int(win_h * (ori_w / ori_h))
    
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Result', win_w, win_h)
    cv2.imshow('Result', warped_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
