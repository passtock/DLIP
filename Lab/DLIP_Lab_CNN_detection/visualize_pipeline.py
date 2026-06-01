import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def visualize_parking_pipeline():
    video_path = 'DLIP_parking_test_video.avi'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video file")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0
        
    # 3분 22초 = 202초
    target_time_sec = (3 * 60) + 22
    target_frame = int(target_time_sec * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Could not read frame at {target_time_sec} seconds")
        return

    # 1. Original ROI
    height, width = frame.shape[:2]
    import json
    import os
    
    roi_config_file = 'roi_config.json'
    if os.path.exists(roi_config_file):
        with open(roi_config_file, 'r') as f:
            roi_data = json.load(f)
            y_start = roi_data['y_start']
            y_end = roi_data['y_end']
        print(f"===== 저장된 ROI 고정값 불러오기 =====")
        print(f"y_start: {y_start}, y_end: {y_end}")
    else:
        print("===== ROI 설정 =====")
        print("마우스로 주차선이 포함될 위아래 영역(높이)을 드래그한 후, ENTER 나 SPACE 를 누르세요.")
        print("창이 뜨면 작업해주세요.")
        # 사용자 인터렉티브 ROI 선택
        roi = cv2.selectROI("Select ROI (Drag and press ENTER)", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI (Drag and press ENTER)")
        
        if roi == (0, 0, 0, 0):
            # 만약 선택하지 않고 닫았다면 기본값 사용
            print("ROI가 선택되지 않아 기본값(0.35 ~ 0.70)을 사용합니다.")
            y_start = int(height * 0.35)
            y_end = int(height * 0.70)
        else:
            x, y, w, h = roi
            y_start = y
            y_end = y + h
            print(f">>> 선택하신 ROI 고정값: y_start={y_start}, y_end={y_end}")
            
        with open(roi_config_file, 'w') as f:
            json.dump({'y_start': y_start, 'y_end': y_end}, f)
        print(f">>> 이 값이 '{roi_config_file}' 에 저장되어 앞으로 자동 고정됩니다!")
        
    roi_img = frame[y_start:y_end, 0:width] # 가로(width)는 항상 전체를 씀
    
    # YOLO 모델 로드 (차량 마스킹용)
    model = YOLO('yolov8n.pt')
    results = model(roi_img, verbose=False)[0]
    
    # 2. Grayscale
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    
    # 3. Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 4. Threshold
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    
    # 5. Masked Threshold (모폴로지 대신 차량을 지우는 단계)
    masked_thresh = thresh.copy()
    for box in results.boxes:
        cls_id = int(box.cls[0])
        # 차량 클래스 필터 (car=2, motorcycle=3, bus=5, truck=7)
        if cls_id in {2, 3, 5, 7}:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 검은색으로 차 영역 덮어쓰기
            cv2.rectangle(masked_thresh, (x1, y1), (x2, y2), 0, -1)
            
    # 6. HoughLines (Masked Threshold에서 선 추출 및 연장)
    lines = cv2.HoughLinesP(masked_thresh, rho=1, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
    hough_img = roi_img.copy()
    roi_h, roi_w = roi_img.shape[:2]
    
    vertical_lines_top_x = []
    vertical_lines_bot_x = []
    
    top_weights = np.zeros(roi_h, dtype=np.float32)
    bot_weights = np.zeros(roi_h, dtype=np.float32)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 기울기에 따라 가로선/세로선을 구분하여 다른 색상으로 표시
            if abs(y2 - y1) > abs(x2 - x1):  # 세로선 (수직에 가까움)
                if y2 != y1:
                    slope = (x2 - x1) / (y2 - y1) # dy에 대한 dx의 변화량
                    x_top = int(x1 + slope * (0 - y1))
                    x_bottom = int(x1 + slope * (roi_h - y1))
                    cv2.line(hough_img, (x_top, 0), (x_bottom, roi_h), (0, 0, 255), 2) # 빨간색
                    vertical_lines_top_x.append(x_top)
                    vertical_lines_bot_x.append(x_bottom)
                else:
                    cv2.line(hough_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    vertical_lines_top_x.append(x1)
                    vertical_lines_bot_x.append(x2)
            else:  # 가로선 (수평에 가까움)
                # 가로선은 선의 길이를 가중치로 주어 가장 긴 선(진짜 주차선)을 동적으로 찾습니다.
                length = abs(x2 - x1)
                y_mid = (y1 + y2) // 2
                if 0 <= y_mid < roi_h:
                    if y_mid < roi_h / 2:
                        for dy in range(-5, 6): # 두께 11픽셀 범위로 가중치 부여
                            if 0 <= y_mid + dy < roi_h:
                                top_weights[y_mid + dy] += length
                    else:
                        for dy in range(-5, 6):
                            if 0 <= y_mid + dy < roi_h:
                                bot_weights[y_mid + dy] += length
                                
    # 가중치가 가장 높은 Y좌표(가장 선명하고 긴 가로선) 2개를 찾습니다.
    top_y = int(np.argmax(top_weights)) if np.max(top_weights) > 0 else int(roi_h * 0.1)
    bot_y = int(np.argmax(bot_weights)) if np.max(bot_weights) > 0 else int(roi_h * 0.9)
    
    # 가로선 2개 파란색으로 화면 끝까지 그리기
    cv2.line(hough_img, (0, top_y), (roi_w, top_y), (255, 0, 0), 2)
    cv2.line(hough_img, (0, bot_y), (roi_w, bot_y), (255, 0, 0), 2)
    
    # 세로선 클러스터링 (비슷한 선들을 하나로 합침)
    vert_pairs = sorted(zip(vertical_lines_top_x, vertical_lines_bot_x), key=lambda p: (p[0] + p[1])/2)
    merged_verts = []
    for p in vert_pairs:
        if not merged_verts:
            merged_verts.append(list(p))
        else:
            prev_p = merged_verts[-1]
            if abs(((prev_p[0]+prev_p[1])/2) - ((p[0]+p[1])/2)) < 25: # 25픽셀 이내면 동일한 선으로 취급하여 평균
                merged_verts[-1][0] = int((prev_p[0] + p[0]) / 2)
                merged_verts[-1][1] = int((prev_p[1] + p[1]) / 2)
            else:
                merged_verts.append(list(p))
                
    # 7. Final Dynamic Result on Full Frame (추출된 선들의 교차점을 기반으로 동적 생성!)
    final_img = frame.copy()
    
    if len(merged_verts) >= 2:
        # 양쪽 끝단 차에 가려져 추출되지 않은 주차선(왼쪽 끝 1개, 오른쪽 끝 1개)을 평균 너비를 이용해 추정(Extrapolation)하여 복원합니다.
        num_lines = len(merged_verts)
        top_mean_width = (merged_verts[-1][0] - merged_verts[0][0]) / (num_lines - 1)
        bot_mean_width = (merged_verts[-1][1] - merged_verts[0][1]) / (num_lines - 1)
        
        # 왼쪽 가상 주차선 추가
        left_bound = [int(merged_verts[0][0] - top_mean_width), int(merged_verts[0][1] - bot_mean_width)]
        merged_verts.insert(0, left_bound)
        
        # 오른쪽 가상 주차선 추가
        right_bound = [int(merged_verts[-1][0] + top_mean_width), int(merged_verts[-1][1] + bot_mean_width)]
        merged_verts.append(right_bound)
        
        for i in range(len(merged_verts) - 1):
            top_left = (merged_verts[i][0], y_start + top_y)
            top_right = (merged_verts[i+1][0], y_start + top_y)
            bot_right = (merged_verts[i+1][1], y_start + bot_y)
            bot_left = (merged_verts[i][1], y_start + bot_y)
            
            poly = np.array([top_left, top_right, bot_right, bot_left], dtype=np.int32)
            cv2.polylines(final_img, [poly], True, (0, 255, 0), 2)

    # Plotting all steps using Matplotlib
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    fig.suptitle('Parking Line Detection Pipeline (3 min 22 sec)', fontsize=16)
    
    def imshow_ax(ax, img, title, cmap=None):
        if cmap is None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')

    imshow_ax(axes[0, 0], frame, '0. Original Frame (3:22)')
    imshow_ax(axes[0, 1], roi_img, '1. ROI Extraction')
    imshow_ax(axes[1, 0], gray, '2. Grayscale', cmap='gray')
    imshow_ax(axes[1, 1], blurred, '3. Gaussian Blur', cmap='gray')
    imshow_ax(axes[2, 0], thresh, '4. Binary Threshold', cmap='gray')
    imshow_ax(axes[2, 1], masked_thresh, '5. Masked Cars (Threshold)', cmap='gray')
    imshow_ax(axes[3, 0], hough_img, '6. HoughLines (No Cars)')
    imshow_ax(axes[3, 1], final_img, '7. Final Hardcoded Spaces (Perfectly Aligned)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('pipeline_visualization.jpg')
    print("Pipeline visualization saved as 'pipeline_visualization.jpg'")
    plt.show() # 화면에 띄워줍니다

if __name__ == "__main__":
    visualize_parking_pipeline()
