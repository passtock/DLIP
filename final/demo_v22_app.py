import streamlit as st
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from PIL import Image
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

# ==========================================
# 1. 모델 아키텍처 정의 (v22_Binary_Endgame)
# ==========================================
CFG = {'full': 512, 'corner': 256, 'edge': 128, 'surface': 384}

class BranchEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.backbone = efficientnet_b4()
        in_feat = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(in_feat, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, x):
        return self.proj(self.pool(self.backbone.features(x)).flatten(1))

class PSABinaryModel(nn.Module):
    def __init__(self, embed_dim=256, dropout=0.4):
        super().__init__()
        self.enc_full = BranchEncoder(embed_dim)
        self.enc_corners = BranchEncoder(embed_dim)
        self.enc_edges = BranchEncoder(embed_dim)
        self.enc_surface = BranchEncoder(embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 4, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=embed_dim*4, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4, norm=nn.LayerNorm(embed_dim))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def _encode_multi(self, encoder, x):
        B, N, C, H, W = x.shape
        out = encoder(x.view(B * N, C, H, W))
        return out.view(B, N, -1).mean(1)

    def forward(self, full, corners, edges, surface):
        t_f = self._encode_multi(self.enc_full, full)
        t_c = self._encode_multi(self.enc_corners, corners)
        t_e = self._encode_multi(self.enc_edges, edges)
        t_s = self._encode_multi(self.enc_surface, surface)
        tokens = torch.stack([t_f, t_c, t_e, t_s], dim=1) + self.pos_embed
        trans_out = self.transformer(tokens)
        return self.head(trans_out.mean(dim=1)).squeeze(1)

# ==========================================
# 2. 전역 설정 및 전처리
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def crop_regions(img: np.ndarray) -> dict:
    H, W = img.shape[:2]
    return {
        "full": cv2.resize(img, (CFG['full'], CFG['full'])),
        "corners": np.stack([cv2.resize(img[0:96, 0:96], (CFG['corner'], CFG['corner'])), cv2.resize(img[0:96, W-96:W], (CFG['corner'], CFG['corner'])), cv2.resize(img[H-96:H, 0:96], (CFG['corner'], CFG['corner'])), cv2.resize(img[H-96:H, W-96:W], (CFG['corner'], CFG['corner']))]),
        "edges": np.stack([cv2.resize(img[0:32, :], (CFG['edge'], CFG['edge'])), cv2.resize(img[H-32:H, :], (CFG['edge'], CFG['edge'])), cv2.resize(img[:, 0:32], (CFG['edge'], CFG['edge'])), cv2.resize(img[:, W-32:W], (CFG['edge'], CFG['edge']))]),
        "surface": cv2.resize(img[H//2-128:H//2+128, W//2-128:W//2+128], (CFG['surface'], CFG['surface']))
    }

eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def process_image_pair(front_img_pil, back_img_pil):
    img_f = np.array(front_img_pil)
    img_b = np.array(back_img_pil)
    rf, rb = crop_regions(img_f), crop_regions(img_b)
    
    apply = lambda img: eval_transform(img)
    
    full = torch.stack([apply(rf["full"]), apply(rb["full"])])
    surface = torch.stack([apply(rf["surface"]), apply(rb["surface"])])
    corners = torch.cat([torch.stack([apply(rf["corners"][i]) for i in range(4)]), torch.stack([apply(rb["corners"][i]) for i in range(4)])])
    edges = torch.cat([torch.stack([apply(rf["edges"][i]) for i in range(4)]), torch.stack([apply(rb["edges"][i]) for i in range(4)])])
    
    # unsqueeze(0) to add batch dimension [1, N, C, H, W]
    return {
        "full": full.unsqueeze(0),
        "surface": surface.unsqueeze(0),
        "corners": corners.unsqueeze(0),
        "edges": edges.unsqueeze(0)
    }

@st.cache_resource
def load_yolo_model():
    """YOLOv8 카드 크롭 모델 로드"""
    yolo_weight_path = r"C:\Users\passp\source\repos\DLIP\Tutorial\yolo\results\detect\train5\weights\best.pt"
    if os.path.exists(yolo_weight_path):
        return YOLO(yolo_weight_path)
    return None

@st.cache_resource
def load_psa_models():
    """v22_Binary_Endgame 5-Fold 앙상블 모델 로드"""
    models = []
    v22_dir = "v22_Binary_Endgame"
    
    if not os.path.exists(v22_dir):
        return models
        
    for i in range(5):
        weight_path = os.path.join(v22_dir, f"best_fold{i}.pth")
        if os.path.exists(weight_path):
            model = PSABinaryModel().to(DEVICE)
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
            model.eval()
            models.append(model)
    return models

def crop_card_with_yolo(image_pil, yolo_model):
    """YOLOv8 모델을 사용해 카드 영역만 크롭"""
    if yolo_model is None:
        return image_pil, False # YOLO가 없으면 원본 리턴
        
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    results = yolo_model(img_cv, verbose=False, conf=0.02, imgsz=640)
    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) > 0:
        x1, y1, x2, y2 = map(int, boxes[0])
        cropped = img_cv[y1:y2, x1:x2]
        return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)), True
    return image_pil, False

def predict_with_tta(model, f, c, e, s):
    probs = []
    probs.append(torch.sigmoid(model(f, c, e, s)))
    probs.append(torch.sigmoid(model(torch.flip(f, [4]), torch.flip(c, [4]), torch.flip(e, [4]), torch.flip(s, [4]))))
    return torch.stack(probs).mean(dim=0)

# ==========================================
# 3. Streamlit UI
# ==========================================
def main():
    st.set_page_config(page_title="PSA 10 AI Predictor", page_icon="🃏", layout="wide")
    
    st.title("🃏 PSA 10 Gem Mint AI Predictor")
    st.markdown("**(v22 Binary Endgame + YOLOv8 Auto Crop)**")
    st.write("카드 앞면과 뒷면 사진을 업로드하면 AI가 자동으로 카드를 찾아 자른 뒤, PSA 10(최고등급)일 예상 확률을 알려줍니다.")
    
    # 모델 로드
    yolo_model = load_yolo_model()
    psa_models = load_psa_models()
    
    if not psa_models:
        st.error("❌ `v22_Binary_Endgame` 폴더 내에 `best_fold0.pth` 등의 모델 가중치를 찾을 수 없습니다.")
        return
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Front (앞면)")
        front_file = st.file_uploader("앞면 이미지 업로드", type=["jpg", "png", "jpeg"], key="front")
        
    with col2:
        st.subheader("Back (뒷면)")
        back_file = st.file_uploader("뒷면 이미지 업로드", type=["jpg", "png", "jpeg"], key="back")
        
    if front_file and back_file:
        st.divider()
        st.subheader("🔍 Analysis Progress")
        
        front_img = Image.open(front_file).convert("RGB")
        back_img = Image.open(back_file).convert("RGB")
        
        with st.spinner("YOLOv8 모델로 카드 영역을 추출하는 중..."):
            front_crop, f_success = crop_card_with_yolo(front_img, yolo_model)
            back_crop, b_success = crop_card_with_yolo(back_img, yolo_model)
            
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(front_crop, caption=f"Front Cropped {'(YOLO)' if f_success else '(Original)'}", use_container_width=True)
        with col_img2:
            st.image(back_crop, caption=f"Back Cropped {'(YOLO)' if b_success else '(Original)'}", use_container_width=True)
            
        with st.spinner(f"v22 앙상블 모델({len(psa_models)} Folds + TTA) 추론 중..."):
            inputs = process_image_pair(front_crop, back_crop)
            f_tensor = inputs["full"].to(DEVICE)
            c_tensor = inputs["corners"].to(DEVICE)
            e_tensor = inputs["edges"].to(DEVICE)
            s_tensor = inputs["surface"].to(DEVICE)
            
            all_probs = []
            with torch.no_grad():
                for m in psa_models:
                    with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                        prob = predict_with_tta(m, f_tensor, c_tensor, e_tensor, s_tensor)
                        all_probs.append(prob)
            
            # 앙상블 평균 (Class 1이 PSA 10)
            final_prob = torch.stack(all_probs).mean().item()
            gem_mint_prob = final_prob * 100
            
        st.divider()
        st.markdown("<h3 style='text-align: center;'>🏆 Prediction Result</h3>", unsafe_allow_html=True)
        
        if gem_mint_prob >= 50.0:
            result_color = "#4CAF50" # Green
            result_text = "GEM MINT (PSA 10) 예상!"
            st.balloons()
        else:
            result_color = "#F44336" # Red
            result_text = "Non-Gem (PSA 8, 9) 예상"
            
        st.markdown(
            f"""
            <div style="text-align: center; margin: 20px 0;">
                <h1 style="font-size: 5rem; font-weight: bold; color: {result_color}; margin-bottom: 0;">{gem_mint_prob:.2f}%</h1>
                <h2 style="margin-top: 10px;">{result_text}</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
            
        st.progress(gem_mint_prob / 100.0)

if __name__ == "__main__":
    main()
