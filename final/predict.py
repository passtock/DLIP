import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageTk, ImageDraw
import timm
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

# ==========================================
# 1. 글로벌 설정
# ==========================================
CFG = {
    'img_size'    : 300,
    'corner_size' : 96,
    'edge_size'   : 32,
    'dropout'     : 0.3
}
MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. 이미지 전처리 및 텐서 변환 모듈
# ==========================================
def make_tf(size):
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return T.Compose([T.Resize((size, size)), T.ToTensor(), norm])

def crop_regions(img):
    W, H = img.size
    cs, ew, eh = CFG['corner_size'], CFG['img_size'], CFG['edge_size']
    return {
        'full': img,
        'tl': img.crop((0, 0, cs, cs)), 'tr': img.crop((W-cs, 0, W, cs)),
        'bl': img.crop((0, H-cs, cs, H)), 'br': img.crop((W-cs, H-cs, W, H)),
        'top': img.crop(((W-ew)//2, 0, (W+ew)//2, eh)), 'bottom': img.crop(((W-ew)//2, H-eh, (W+ew)//2, H)),
        'left': img.crop((0, (H-ew)//2, eh, (H+ew)//2)), 'right': img.crop((W-eh, (H-ew)//2, W, (H+ew)//2)),
        'surface': img.crop((W//4, H//4, 3*W//4, 3*H//4)),
    }

def preprocess_card(front_img, back_img):
    tf_full = make_tf(CFG['img_size'])
    tf_corner = make_tf(CFG['corner_size'])
    tf_edge = make_tf(CFG['edge_size'])
    tf_surface = make_tf(CFG['img_size']//2)
    
    cf = crop_regions(front_img)
    cb = crop_regions(back_img)

    full = torch.cat([tf_full(cf['full']), tf_full(cb['full'])], dim=0)
    corners = torch.stack([
        tf_corner(cf['tl']), tf_corner(cf['tr']), tf_corner(cf['bl']), tf_corner(cf['br']),
        tf_corner(cb['tl']), tf_corner(cb['tr']), tf_corner(cb['bl']), tf_corner(cb['br'])
    ]).reshape(24, CFG['corner_size'], CFG['corner_size'])
    edges = torch.stack([
        tf_edge(cf['top']), tf_edge(cf['bottom']), tf_edge(cf['left']), tf_edge(cf['right']),
        tf_edge(cb['top']), tf_edge(cb['bottom']), tf_edge(cb['left']), tf_edge(cb['right'])
    ]).reshape(24, CFG['edge_size'], CFG['edge_size'])
    surface = torch.cat([tf_surface(cf['surface']), tf_surface(cb['surface'])], dim=0)

    return full.unsqueeze(0).to(DEVICE), corners.unsqueeze(0).to(DEVICE), edges.unsqueeze(0).to(DEVICE), surface.unsqueeze(0).to(DEVICE)

# ==========================================
# 3. 드래그 UI 및 고해상도 크롭 엔진
# ==========================================
def auto_detect_bounding_box(pil_img):
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 30, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area > (cv_img.shape[0] * cv_img.shape[1] * 0.15):
            return (x, y, x+w, y+h)
    return None

class DragCropSelector(tk.Toplevel):
    def __init__(self, parent, pil_img, is_slab=True):
        super().__init__(parent)
        self.crop_box = None
        self.title("마우스 드래그 영역 지정 (1:1.4 비율 고정)")
        
        # 🚀 크롭 화면 1.5배 확대 (1200x900)
        max_w, max_h = 1200, 900
        w, h = pil_img.size
        self.scale = min(max_w / w, max_h / h, 1.0)
        self.display_w, self.display_h = int(w * self.scale), int(h * self.scale)
        
        resized = pil_img.resize((self.display_w, self.display_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)
        
        guide_text = "마우스로 드래그하여 [남길 카드 영역]을 지정하세요.\n(스포츠 카드 표준 비율 1:1.4로 자동 고정되어 왜곡을 방지합니다.)"
        tk.Label(self, text=guide_text, font=("Arial", 11, "bold"), fg="#c0392b" if is_slab else "#2980b9").pack(pady=10)
        
        self.canvas = tk.Canvas(self, width=self.display_w, height=self.display_h, bg="black", cursor="cross")
        self.canvas.pack(padx=10, pady=10)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        
        self.rect_id = None
        self.start_x = None
        self.start_y = None
        
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.grab_set()
        self.wait_window()
        
    def on_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='#2ecc71', width=3, dash=(4, 2))
        
    def on_drag(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        
        dx = cur_x - self.start_x
        dy = cur_y - self.start_y
        
        # 🚀 드래그 시 가로세로 비율 1.4 고정
        ratio = 1.4
        if abs(dx) * ratio > abs(dy):
            dy = abs(dx) * ratio * (1 if dy > 0 else -1)
            cur_y = self.start_y + dy
        else:
            dx = (abs(dy) / ratio) * (1 if dx > 0 else -1)
            cur_x = self.start_x + dx
            
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)
        
    def on_release(self, event):
        if not self.rect_id: return
        coords = self.canvas.coords(self.rect_id)
        
        if abs(coords[2] - coords[0]) < 20:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
            return
            
        x1, y1 = min(coords[0], coords[2]) / self.scale, min(coords[1], coords[3]) / self.scale
        x2, y2 = max(coords[0], coords[2]) / self.scale, max(coords[1], coords[3]) / self.scale
        
        self.crop_box = (int(x1), int(y1), int(x2), int(y2))
        self.destroy()

# ==========================================
# 4. 동적 두께 적용 HUD 레이어
# ==========================================
def draw_defect_boxes(img, branch_idx):
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    W, H = img.size
    cs, ew, eh = CFG['corner_size'], CFG['img_size'], CFG['edge_size']
    
    line_width = max(4, int(W * 0.008))
    color = "#e74c3c"
    
    if branch_idx == 0:  # Centering
        draw.rectangle([10, 10, W-10, H-10], outline=color, width=line_width)
        m, L = 50, 40
        for px, py in [(m, m), (W-m, m), (m, H-m), (W-m, H-m)]:
            sx = -1 if px > W/2 else 1
            sy = -1 if py > H/2 else 1
            draw.line([px, py, px + L*sx, py], fill=color, width=line_width)
            draw.line([px, py, px, py + L*sy], fill=color, width=line_width)
    elif branch_idx == 1:  # Corners
        for x1, y1, x2, y2 in [(0, 0, cs, cs), (W-cs, 0, W, cs), (0, H-cs, cs, H), (W-cs, H-cs, W, H)]:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
    elif branch_idx == 2:  # Edges
        draw.rectangle([max(0, (W-ew)//2), 0, min(W, (W+ew)//2), eh], outline=color, width=line_width)
        draw.rectangle([max(0, (W-ew)//2), H-eh, min(W, (W+ew)//2), H], outline=color, width=line_width)
        draw.rectangle([0, max(0, (H-ew)//2), eh, min(H, (H+ew)//2)], outline=color, width=line_width)
        draw.rectangle([W-eh, max(0, (H-ew)//2), W, min(H, (H+ew)//2)], outline=color, width=line_width)
    elif branch_idx == 3:  # Surface
        draw.rectangle([W//4, H//4, 3*W//4, 3*H//4], outline=color, width=line_width)
    return annotated

# ==========================================
# 5. 앙상블 모델 로더
# ==========================================
class RegionEncoder(nn.Module):
    def __init__(self, in_channels, out_dim=128):
        super().__init__()
        base = timm.create_model('efficientnet_b2', pretrained=True)
        old = base.conv_stem
        new_conv = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            for i in range(in_channels): new_conv.weight[:, i] = old.weight[:, i % 3] / (in_channels / 3)
        base.conv_stem = new_conv
        base.classifier = nn.Sequential(nn.Linear(base.classifier.in_features, out_dim), nn.ReLU())
        self.encoder = base
    def forward(self, x): return self.encoder(x)

class AttentionFusion(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        return self.proj(self.attn_drop(attn.softmax(dim=-1)) @ v)

class PSAMultiBranchModel(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.centering = RegionEncoder(6, embed_dim)
        self.corner = RegionEncoder(24, embed_dim)
        self.edge = RegionEncoder(24, embed_dim)
        self.surface = RegionEncoder(6, embed_dim)
        self.attention_fusion = AttentionFusion(dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(CFG['dropout']), nn.Linear(embed_dim * 4, 128),
            nn.GELU(), nn.Dropout(CFG['dropout']/2), nn.Linear(128, 2)
        )
    def forward(self, f, c, e, s, mask_idx=None):
        features = torch.stack([self.centering(f), self.corner(c), self.edge(e), self.surface(s)], dim=1)
        if mask_idx is not None: features[:, mask_idx, :] = 0.0
        return self.classifier(self.attention_fusion(features).reshape(features.size(0), -1))

import warnings
warnings.filterwarnings("ignore")
print(f"[{DEVICE}] 구조 모델 가중치 파일 로드 시작...")
models = []
for fold in range(1, 6):
    weight_path = os.path.join(MODEL_DIR, f'best_model_fold{fold}.pth')
    if os.path.exists(weight_path):
        m = PSAMultiBranchModel().to(DEVICE)
        m.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        m.eval()
        models.append(m)

# ==========================================
# 6. GUI 애플리케이션
# ==========================================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("PSA Grading Predictor (최종 실전 인퍼런스 앱)")
        self.root.geometry("820x720")
        
        self.front_path, self.back_path = "", ""
        self.orig_front, self.orig_back = None, None
        
        tk.Label(root, text="PSA 스포츠 카드 예상 등급 판정 시스템", font=("Arial", 14, "bold")).pack(pady=10)
        
        mode_frame = tk.LabelFrame(root, text=" ⚙️ 검사 모드 선택 ", font=("Arial", 10, "bold"), padx=15, pady=5)
        mode_frame.pack(pady=5)
        
        self.mode_var = tk.StringVar(value="manual")
        tk.Radiobutton(mode_frame, text="수동 정밀 모드 (마우스로 직접 영역 1:1.4 드래그)", variable=self.mode_var, value="manual", font=("Arial", 10), fg="#8e44ad").grid(row=0, column=0, padx=15)
        tk.Radiobutton(mode_frame, text="생카드 (자동 박스 탐지 - 배경이 어두울 때만 권장)", variable=self.mode_var, value="raw", font=("Arial", 10)).grid(row=0, column=1, padx=15)
        
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        
        self.btn_front = tk.Button(btn_frame, text="1. 앞면 사진 넣기", command=self.select_front, width=22, bg="#34495e", fg="white", font=("Arial", 10))
        self.btn_front.grid(row=0, column=0, padx=10, pady=5)
        
        self.btn_back = tk.Button(btn_frame, text="2. 뒷면 사진 넣기", command=self.select_back, width=22, bg="#34495e", fg="white", font=("Arial", 10))
        self.btn_back.grid(row=0, column=1, padx=10, pady=5)
        
        self.btn_run = tk.Button(btn_frame, text="🚀 예상 등급 및 결함 판정", command=self.run_inference, width=25, bg="#e74c3c", fg="white", font=("Arial", 11, "bold"))
        self.btn_run.grid(row=1, column=0, padx=10, pady=10)
        
        self.btn_reset = tk.Button(btn_frame, text="🔄 다시하기 (Reset)", command=self.reset_all, width=22, bg="#7f8c8d", fg="white", font=("Arial", 10, "bold"))
        self.btn_reset.grid(row=1, column=1, padx=10, pady=10)
        
        self.img_frame = tk.Frame(root, bd=2, relief="groove", bg="#f8f9fa")
        self.img_frame.pack(expand=True, fill="both", padx=20, pady=5)
        
        self.lbl_img_front = tk.Label(self.img_frame, text="[ 앞면 대기 중 ]", fg="gray", bg="#f8f9fa", width=35, height=15)
        self.lbl_img_front.pack(side="left", expand=True, fill="both", padx=10, pady=10)
        
        self.lbl_img_back = tk.Label(self.img_frame, text="[ 뒷면 대기 중 ]", fg="gray", bg="#f8f9fa", width=35, height=15)
        self.lbl_img_back.pack(side="right", expand=True, fill="both", padx=10, pady=10)
        
        self.lbl_status = tk.Label(root, text="사진을 불러오면 크롭 창이 뜹니다. 넉넉하고 반듯하게 카드를 잘라주세요.", fg="gray", font=("Arial", 10))
        self.lbl_status.pack(pady=10)

    def resize_to_thumbnail(self, pil_img, max_size=(280, 360)):
        img = pil_img.copy()
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)

    def process_image_by_mode(self, path, side_name):
        raw_img = Image.open(path).convert('RGB')
        current_mode = self.mode_var.get()
        
        if current_mode == "raw":
            auto_box = auto_detect_bounding_box(raw_img)
            if auto_box is not None:
                # 🚀 리사이즈 없이 원본 해상도 그대로 전달 (스케일 보존)
                return raw_img.crop(auto_box)
                
        selector = DragCropSelector(self.root, raw_img, is_slab=False)
        if selector.crop_box:
            # 🚀 리사이즈 없이 드래그한 해상도 그대로 전달 (스케일 보존)
            return raw_img.crop(selector.crop_box)
        return None

    def select_front(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            processed = self.process_image_by_mode(path, "앞면")
            if processed:
                self.orig_front, self.front_path = processed, path
                self.btn_front.config(text="앞면 크롭 완료 ✔", bg="#2ecc71")
                self.tk_front = self.resize_to_thumbnail(self.orig_front)
                self.lbl_img_front.config(image=self.tk_front, text="")

    def select_back(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            processed = self.process_image_by_mode(path, "뒷면")
            if processed:
                self.orig_back, self.back_path = processed, path
                self.btn_back.config(text="뒷면 크롭 완료 ✔", bg="#2ecc71")
                self.tk_back = self.resize_to_thumbnail(self.orig_back)
                self.lbl_img_back.config(image=self.tk_back, text="")

    def run_prediction_core(self, mask_idx=None):
        if len(models) == 0: return 0.0
        f, c, e, s = preprocess_card(self.orig_front, self.orig_back)
        
        probs_list = []
        with torch.no_grad():
            for m in models:
                # 🚀 헷갈리게 만들던 상하 반전 검사(TTA) 완전 제거. 정방향 1회만 테스트
                out = m(f, c, e, s, mask_idx=mask_idx)
                probs_list.append(F.softmax(out, dim=1))
                
        return torch.stack(probs_list).mean(0).cpu().numpy()[0][1]

    def run_inference(self):
        if not self.front_path or not self.back_path:
            messagebox.showwarning("경고", "양면 사진 크롭이 모두 완료되어야 판정이 가능합니다.")
            return
        self.lbl_status.config(text="AI 모델이 결함을 정밀 분석하고 있습니다...", fg="blue")
        self.root.update()
        
        try:
            base_score = self.run_prediction_core()
            branch_names = ['중심도 비율 (Centering)', '모서리 훼손 (Corners)', '가장자리 테두리 (Edges)', '표면 스크래치/오염 (Surface)']
            gains = [max(0.0, self.run_prediction_core(mask_idx=i) - base_score) for i in range(4)]
            
            total_gain = sum(gains)
            contributions = [(g / total_gain) * 100 if total_gain > 0 else 0.0 for g in gains]
            
            analysis_report = "\n🔍 [AI 부위별 감점 기여도 보고서]\n"
            for name, pct in zip(branch_names, contributions):
                bar = "🔥" * int(round(pct / 10)) if pct > 0 else "⬜"
                analysis_report += f"- {name}: {pct:.1f}% ({bar if bar else '🔥'})\n"
            
            max_idx = np.argmax(gains) if total_gain > 0 else None
            if max_idx is not None and base_score < 0.98:
                self.tk_front = self.resize_to_thumbnail(draw_defect_boxes(self.orig_front, max_idx))
                self.tk_back = self.resize_to_thumbnail(draw_defect_boxes(self.orig_back, max_idx))
                self.lbl_img_front.config(image=self.tk_front)
                self.lbl_img_back.config(image=self.tk_back)
                culprit_text = f"\n⚠️ 가장 큰 결함 의심 부위: {branch_names[max_idx]}"
            else:
                culprit_text = ""

            # 🚀 기준점(Threshold) 55%로 조정 완료
            if base_score >= 0.65:
                self.lbl_status.config(text="결과: ✨ Gem Mint 10 확정수준", fg="green", font=("Arial", 11, "bold"))
                messagebox.showinfo("최종 검사 결과", f"완벽한 상태입니다!\n\nAI 10등급 확신도: {base_score*100:.2f}%\n\n{analysis_report}")
            elif base_score >= 0.55:
                self.lbl_status.config(text="결과: ✅ Gem Mint 10 획득 유력", fg="#d35400", font=("Arial", 11, "bold"))
                messagebox.showinfo("최종 검사 결과", f"제출을 적극 권장합니다.\n\nAI 10등급 확신도: {base_score*100:.2f}%\n\n{analysis_report}{culprit_text}")
            elif base_score >= 0.50: 
                self.lbl_status.config(text="결과: ⚠️ Borderline (9.5등급 / 재검토 요망)", fg="#e67e22", font=("Arial", 11, "bold"))
                messagebox.showinfo("최종 검사 결과", f"10등급과 9등급의 아슬아슬한 경계선에 있습니다.\n\nAI 10등급 확신도: {base_score*100:.2f}%\n\n{analysis_report}{culprit_text}")
            else:
                self.lbl_status.config(text="결과: ❌ Non-Gem (8, 9등급 구역 판정)", fg="red", font=("Arial", 11, "bold"))
                messagebox.showinfo("최종 검사 결과", f"결함이 감지되어 8, 9등급 이하 구역으로 판정되었습니다.\n\nAI 10등급 확신도: {base_score*100:.2f}%\n\n{analysis_report}{culprit_text}")
        except Exception as e:
            messagebox.showerror("오류", f"연산 에러: {str(e)}")

    def reset_all(self):
        self.front_path, self.back_path = "", ""
        self.orig_front, self.orig_back, self.tk_front, self.tk_back = None, None, None, None
        self.btn_front.config(text="1. 앞면 사진 넣기", bg="#34495e")
        self.btn_back.config(text="2. 뒷면 사진 넣기", bg="#34495e")
        self.lbl_img_front.config(image="", text="[ 앞면 대기 중 ]")
        self.lbl_img_back.config(image="", text="[ 뒷면 대기 중 ]")
        self.lbl_status.config(text="사진을 불러오면 크롭 창이 뜹니다. 넉넉하고 반듯하게 카드를 잘라주세요.", fg="gray", font=("Arial", 10))

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()