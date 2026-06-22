import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4, resnet18
import torchvision.transforms as T
import sys

# ==========================================
# 1. Model Architecture (Must match Training)
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
            nn.Linear(1024, 1)
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
# 2. Grad-CAM Implementation
# ==========================================
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.features = None
        self.gradients = None
        
        target_layer = self.model.efficientnet.features[-1]
        target_layer.register_forward_hook(self.save_features)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_features(self, module, input, output):
        self.features = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, full_img, tl, tr, bl, br):
        self.model.eval()
        output = self.model(full_img, tl, tr, bl, br)
        
        self.model.zero_grad()
        output.backward(retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.features, dim=1).squeeze(0)
        cam = F.relu(cam) 
        
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        prob_10 = torch.sigmoid(output).item() * 100
        return cam.detach().cpu().numpy(), prob_10

# ==========================================
# 3. Centering Analysis (OpenCV)
# ==========================================
def analyze_centering(warped_img):
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_area = 1000 * 1400
    best_bb = None
    max_area = 0
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / float(h + 1e-5) 
        
        if area > 200000 and area < img_area * 0.90: 
            if 0.55 < aspect_ratio < 0.85:
                if area > max_area:
                    max_area = area
                    best_bb = (x, y, w, h)
                
    warnings = []
    if best_bb is not None:
        x, y, w, h = best_bb
        left_gap = x
        right_gap = 1000 - (x + w)
        top_gap = y
        bottom_gap = 1400 - (y + h)
        
        lr_ratio = (left_gap / (left_gap + right_gap + 1e-5)) * 100
        tb_ratio = (top_gap / (top_gap + bottom_gap + 1e-5)) * 100
        
        lr_str = f"{lr_ratio:.1f} / {100-lr_ratio:.1f}"
        tb_str = f"{tb_ratio:.1f} / {100-tb_ratio:.1f}"
        
        if not (40 <= lr_ratio <= 60):
            warnings.append(f"L/R Centering Off ({lr_str})")
        if not (40 <= tb_ratio <= 60):
            warnings.append(f"T/B Centering Off ({tb_str})")
            
        return lr_str, tb_str, warnings, best_bb
        
    return "N/A", "N/A", ["Failed to detect inner border"], None

# ==========================================
# 4. Image Processing & Auto-Crop
# ==========================================
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def process_image(img_path, manual_crop=False):
    img = cv2.imread(img_path)
    if img is None:
        print("[ERROR] Image not found.")
        sys.exit(1)
        
    ori_h, ori_w = img.shape[:2]
    pts = None
    
    if manual_crop:
        display_img = img.copy()
        win_h = 800
        win_w = int(win_h * (ori_w / ori_h))
        cv2.namedWindow('Select Card ROI', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Select Card ROI', win_w, win_h)
        print("[INFO] Please drag to select the card area, then press SPACE or ENTER.")
        roi = cv2.selectROI('Select Card ROI', display_img, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        if w != 0 and h != 0:
            pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype="float32")
            
    if pts is None:
        print("[INFO] Attempting Auto-Crop via Contour Detection...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype("float32")
                break
                
        if pts is None:
            print("[WARNING] Auto-Crop failed. Using full image bounds.")
            pts = np.array([[0, 0], [ori_w, 0], [ori_w, ori_h], [0, ori_h]], dtype="float32")
            
    rect = order_points(pts)
    dst_w, dst_h = 1000, 1400
    dst = np.array([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (dst_w, dst_h))
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    patch_size = 300
    patches = {
        "Full": cv2.resize(warped_rgb, (512, 512)),
        "TL": warped_rgb[0:patch_size, 0:patch_size],
        "TR": warped_rgb[0:patch_size, dst_w-patch_size:dst_w],
        "BL": warped_rgb[dst_h-patch_size:dst_h, 0:patch_size],
        "BR": warped_rgb[dst_h-patch_size:dst_h, dst_w-patch_size:dst_w]
    }
    
    eval_transform = T.Compose([
        T.ToPILImage(), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return {k: eval_transform(v).unsqueeze(0) for k, v in patches.items()}, warped

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_WEIGHTS = os.path.join(SCRIPT_DIR, "best_psa_binary_model.pth") 
    TEST_IMAGE = os.path.join(SCRIPT_DIR, "test_card.jpg")
    
    print("\n[SETUP] Select Image Processing Mode:")
    print("1 : Manual Crop (Drag ROI)")
    print("2 : Auto-Crop (Contour Detection)")
    choice = input("Enter choice (1/2) >> ").strip()
    is_manual = True if choice == '1' else False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Initializing Processor on {device}...")
    
    model = GradingBinaryModel().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"[ERROR] Weights file not found at {MODEL_WEIGHTS}")
        sys.exit(1)
        
    inputs, warped_bgr = process_image(TEST_IMAGE, manual_crop=is_manual)
    
    print("[INFO] Running AI Inference & Grad-CAM...")
    cam_extractor = GradCAM(model)
    full = inputs["Full"].to(device)
    tl, tr = inputs["TL"].to(device), inputs["TR"].to(device)
    bl, br = inputs["BL"].to(device), inputs["BR"].to(device)
    
    cam_gray, prob_10 = cam_extractor.generate_cam(full, tl, tr, bl, br)
    
    print("[INFO] Analyzing Centering...")
    lr_str, tb_str, c_warnings, best_bb = analyze_centering(warped_bgr)
    
    prob_not_10 = 100 - prob_10
    final_pred = "PSA 10 (Gem Mint)" if prob_10 >= 50.0 else "Not 10 (PSA 8/9)"
    
    print("\n========================================")
    print(" FINAL DIAGNOSTIC REPORT ")
    print("========================================")
    print(f"[ AI Classification ]")
    print(f"  - PSA 10 Probability : {prob_10:.2f}%")
    print(f"  - Defect Probability : {prob_not_10:.2f}%")
    print(f"[ Centering Measurements ]")
    print(f"  - L/R Ratio : {lr_str}")
    print(f"  - T/B Ratio : {tb_str}")
    
    print("\n[ Risk Factors ]")
    if c_warnings:
        for w in c_warnings:
            print(f"  - WARNING: {w}")
    if prob_10 < 50.0:
        print("  - WARNING: AI detected surface/corner defects (See heatmap).")
    
    if not c_warnings and prob_10 >= 50.0:
        print("  - PASS: No significant issues detected.")
         
    print(f"\nFINAL DECISION: {final_pred}")
    
    cam_resized = cv2.resize(cam_gray, (1000, 1400))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(warped_bgr, 0.6, heatmap, 0.4, 0)
    
    if best_bb is not None:
        cv2.rectangle(overlay, (best_bb[0], best_bb[1]), 
                      (best_bb[0]+best_bb[2], best_bb[1]+best_bb[3]), (0, 255, 0), 3)
                      
    color = (0, 255, 0) if prob_10 >= 50.0 else (0, 0, 255)
    cv2.putText(overlay, f"{final_pred}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5)
    cv2.putText(overlay, f"Probability: {prob_10:.1f}%", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    
    win_h = 800
    win_w = int(win_h * (1000 / 1400))
    cv2.namedWindow('XAI Defect Analysis', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('XAI Defect Analysis', win_w, win_h)
    cv2.imshow('XAI Defect Analysis', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()