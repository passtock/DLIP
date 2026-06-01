from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "Lab" / "DLIP_Lab_CNN_detection" / "DLIP_parking_test_video.avi"
DEFAULT_MODEL = "yolov8n-seg.pt" # Updated to YOLOv8 Segmentation model
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_VIDEO = DEFAULT_OUTPUT_DIR / "parking_result.mp4"
DEFAULT_COUNT_FILE = DEFAULT_OUTPUT_DIR / "counting_22000561.txt"

VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # COCO dataset: car, motorcycle, bus, truck
CENTER_Y_CORRECTION = 10 # Correction factor mentioned in the video logic

# ROI configuration (Tuned for the specific Han-dong Univ test video)
ROI_Y_START = 0.45
ROI_Y_END = 0.65


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parking Space Detection System based on Video Algorithm")
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE), help="Input video path")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="YOLO model path")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output folder")
    parser.add_argument("--count-file", type=str, default=str(DEFAULT_COUNT_FILE), help="Result txt file")
    parser.add_argument("--result-video", type=str, default=str(DEFAULT_RESULT_VIDEO), help="Result mp4 file")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--max-spaces", type=int, default=13, help="Number of parking spaces to detect")
    return parser.parse_args()


def get_parking_spots(frame: np.ndarray, max_spaces: int) -> list[np.ndarray]:
    import json
    import os
    from ultralytics import YOLO
    
    height, width = frame.shape[:2]
    roi_config_file = os.path.join(os.path.dirname(__file__), 'roi_config.json')
    if os.path.exists(roi_config_file):
        with open(roi_config_file, 'r') as f:
            roi_data = json.load(f)
            y_start = roi_data['y_start']
            y_end = roi_data['y_end']
    else:
        y_start = int(height * 0.35)
        y_end = int(height * 0.70)
        
    roi_img = frame[y_start:y_end, 0:width]
    
    model = YOLO('yolov8n.pt')
    results = model(roi_img, verbose=False)[0]
    
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    
    masked_thresh = thresh.copy()
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in {2, 3, 5, 7}:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(masked_thresh, (x1, y1), (x2, y2), 0, -1)
            
    lines = cv2.HoughLinesP(masked_thresh, rho=1, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
    
    roi_h, roi_w = roi_img.shape[:2]
    vertical_lines_top_x = []
    vertical_lines_bot_x = []
    top_weights = np.zeros(roi_h, dtype=np.float32)
    bot_weights = np.zeros(roi_h, dtype=np.float32)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) > abs(x2 - x1):
                if y2 != y1:
                    slope = (x2 - x1) / (y2 - y1)
                    x_top = int(x1 + slope * (0 - y1))
                    x_bottom = int(x1 + slope * (roi_h - y1))
                    vertical_lines_top_x.append(x_top)
                    vertical_lines_bot_x.append(x_bottom)
                else:
                    vertical_lines_top_x.append(x1)
                    vertical_lines_bot_x.append(x2)
            else:
                length = abs(x2 - x1)
                y_mid = (y1 + y2) // 2
                if 0 <= y_mid < roi_h:
                    if y_mid < roi_h / 2:
                        for dy in range(-5, 6):
                            if 0 <= y_mid + dy < roi_h:
                                top_weights[y_mid + dy] += length
                    else:
                        for dy in range(-5, 6):
                            if 0 <= y_mid + dy < roi_h:
                                bot_weights[y_mid + dy] += length
                                
    top_y = int(np.argmax(top_weights)) if np.max(top_weights) > 0 else int(roi_h * 0.1)
    bot_y = int(np.argmax(bot_weights)) if np.max(bot_weights) > 0 else int(roi_h * 0.9)
    
    vert_pairs = sorted(zip(vertical_lines_top_x, vertical_lines_bot_x), key=lambda p: (p[0] + p[1])/2)
    merged_verts = []
    for p in vert_pairs:
        if not merged_verts:
            merged_verts.append(list(p))
        else:
            prev_p = merged_verts[-1]
            if abs(((prev_p[0]+prev_p[1])/2) - ((p[0]+p[1])/2)) < 25:
                merged_verts[-1][0] = int((prev_p[0] + p[0]) / 2)
                merged_verts[-1][1] = int((prev_p[1] + p[1]) / 2)
            else:
                merged_verts.append(list(p))
                
    slots = []
    if len(merged_verts) >= 2:
        num_lines = len(merged_verts)
        top_mean_width = (merged_verts[-1][0] - merged_verts[0][0]) / (num_lines - 1)
        bot_mean_width = (merged_verts[-1][1] - merged_verts[0][1]) / (num_lines - 1)
        
        left_bound = [int(merged_verts[0][0] - top_mean_width), int(merged_verts[0][1] - bot_mean_width)]
        merged_verts.insert(0, left_bound)
        
        right_bound = [int(merged_verts[-1][0] + top_mean_width), int(merged_verts[-1][1] + bot_mean_width)]
        merged_verts.append(right_bound)
        
        for i in range(len(merged_verts) - 1):
            top_left = (merged_verts[i][0], y_start + top_y)
            top_right = (merged_verts[i+1][0], y_start + top_y)
            bot_right = (merged_verts[i+1][1], y_start + bot_y)
            bot_left = (merged_verts[i][1], y_start + bot_y)
            
            slot_polygon = np.array([top_left, top_right, bot_right, bot_left], dtype=np.int32)
            slots.append(shrink_polygon(slot_polygon, scale=0.85))
            
    return slots


def shrink_polygon(polygon: np.ndarray, scale: float) -> np.ndarray:
    centroid = polygon.mean(axis=0)
    shrunk = centroid + (polygon.astype(np.float32) - centroid) * scale
    return np.round(shrunk).astype(np.int32)


def get_vehicle_detections(result, conf_threshold: float) -> list[dict]:
    detections = []
    if result.boxes is None or result.masks is None:
        return detections

    for box, mask_xy in zip(result.boxes, result.masks.xy):
        class_id = int(box.cls.item())
        if class_id not in VEHICLE_CLASS_IDS:
            continue

        confidence = float(box.conf.item())
        if confidence < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        detections.append({
            "confidence": confidence,
            "polygon": np.array(mask_xy, dtype=np.int32),
            "center": (center_x, center_y)
        })
    return detections


def check_occupancy(detections: list[dict], parking_spots: list[np.ndarray], frame_shape: tuple) -> list[bool]:
    """Checks if the vehicle segmentation masks cover at least 30% of a parking slot."""
    occupied_status = [False] * len(parking_spots)
    if not detections:
        return occupied_status
        
    height, width = frame_shape[:2]
    
    # 1. Create a combined mask for all detected vehicles
    vehicle_mask = np.zeros((height, width), dtype=np.uint8)
    for det in detections:
        cv2.fillPoly(vehicle_mask, [det["polygon"]], 255)
        
    # 2. Check overlap for each parking spot
    for idx, spot_polygon in enumerate(parking_spots):
        spot_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(spot_mask, [spot_polygon], 255)
        
        spot_area = cv2.countNonZero(spot_mask)
        if spot_area == 0:
            continue
            
        intersection = cv2.bitwise_and(spot_mask, vehicle_mask)
        overlap_area = cv2.countNonZero(intersection)
        
        if (overlap_area / spot_area) >= 0.30:
            occupied_status[idx] = True
            
    return occupied_status


def draw_visuals(frame: np.ndarray, spots: list[np.ndarray], occupancy: list[bool], detections: list[dict]):
    occupied_count = 0
    total_spaces = len(spots)
    
    # Draw parking slots
    for idx, (polygon, is_occupied) in enumerate(zip(spots, occupancy)):
        if is_occupied:
            occupied_count += 1
            color = (0, 0, 255) # Red for occupied
            label = f"{idx + 1}"
        else:
            color = (0, 255, 0) # Green for empty
            label = f"{idx + 1}"
            
        cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)
        
        # Draw Slot Number
        center_x = int(np.mean(polygon[:, 0]))
        center_y = int(np.mean(polygon[:, 1])) - 15
        cv2.putText(frame, label, (center_x - 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    # Draw Vehicle Bounding Boxes and corrected centers
    for det in detections:
        cx, cy = det["center"]
        # Draw the corrected center point used for logic
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1) 
        
    # Draw Summary Info
    empty_spaces = max(0, total_spaces - occupied_count)
    info_text = f"Available Space: {empty_spaces}"
    cv2.putText(frame, info_text, (frame.shape[1] - 300, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(frame, info_text, (frame.shape[1] - 300, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)

    return occupied_count


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load YOLO model
    model = YOLO(args.model)
    
    # Initialize Video Capture
    source_val = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source_val)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.result_video, fourcc, fps, (width, height))
    
    # Setup initial parking spots based on a very clean frame (3 min 22 sec)
    target_time_sec = 3 * 60 + 22
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_time_sec * fps))
    success, ref_frame = cap.read()
    if not success:
        print("Failed to read reference frame.")
        return
        
    parking_spots = get_parking_spots(ref_frame, args.max_spaces)
    
    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_idx = 0
    with open(args.count_file, 'w') as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx >= 1500:
                print("Reached 1500 frames limit. Stopping.")
                break
                
            # 1. Run YOLO inference
            results = model.predict(frame, verbose=False)
            
            # 2. Extract detections and correct centers
            detections = get_vehicle_detections(results[0], args.conf)
            
            # 3. Check which slots are occupied (30% area logic)
            occupancy = check_occupancy(detections, parking_spots, frame.shape)
            
            # 4. Draw visualizations (Use YOLO's segmentation plotter without boxes)
            annotated_frame = results[0].plot(boxes=False)
            occupied_count = draw_visuals(annotated_frame, parking_spots, occupancy, detections)
            
            # 5. Save results
            empty_spaces = len(parking_spots) - occupied_count
            f.write(f"{frame_idx},{empty_spaces}\n")
            out.write(annotated_frame)
            
            # Display
            preview = cv2.resize(annotated_frame, (1024, 768))
            cv2.imshow("Parking Detection System", preview)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing finished.")

if __name__ == "__main__":
    main()