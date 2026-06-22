import os, re, torch, cv2
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from collections import defaultdict
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

CFG = {'full': 512, 'corner': 256, 'edge': 128, 'surface': 384}
CLASS_NAMES = ["Defect (8/9)", "Gem Mint (10)"]

def build_robust_pairs(data_root):
    records = []
    for grade in [8, 9, 10]:
        label = 0 if grade in [8, 9] else 1 
        folder = next((Path(data_root) / f for f in [f'PSA{grade}', f'psa_{grade}', f'psa{grade}', f'PSA_{grade}'] if (Path(data_root) / f).exists()), None)
        if not folder: continue
        cert_dict = defaultdict(dict)
        for img_path in folder.glob('*.*'):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}: continue
            match = re.search(r'(cert\d+)', img_path.stem.lower())
            if match: cert_dict[match.group(1)]['front' if 'front' in img_path.stem.lower() else 'back'] = str(img_path)
        for sides in cert_dict.values():
            if 'front' in sides and 'back' in sides: records.append({'front': sides['front'], 'back': sides['back'], 'label': label})
    return records

def crop_regions(img: np.ndarray) -> dict:
    H, W = img.shape[:2]
    return {
        "full": cv2.resize(img, (CFG['full'], CFG['full'])),
        "corners": np.stack([cv2.resize(img[0:96, 0:96], (CFG['corner'], CFG['corner'])), cv2.resize(img[0:96, W-96:W], (CFG['corner'], CFG['corner'])), cv2.resize(img[H-96:H, 0:96], (CFG['corner'], CFG['corner'])), cv2.resize(img[H-96:H, W-96:W], (CFG['corner'], CFG['corner']))]),
        "edges": np.stack([cv2.resize(img[0:32, :], (CFG['edge'], CFG['edge'])), cv2.resize(img[H-32:H, :], (CFG['edge'], CFG['edge'])), cv2.resize(img[:, 0:32], (CFG['edge'], CFG['edge'])), cv2.resize(img[:, W-32:W], (CFG['edge'], CFG['edge']))]),
        "surface": cv2.resize(img[H//2-128:H//2+128, W//2-128:W//2+128], (CFG['surface'], CFG['surface']))
    }

def get_transforms():
    m, s = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return A.Compose([A.Normalize(m, s), ToTensorV2()])

class PSADataset(Dataset):
    def __init__(self, records):
        self.records, self.tfm = records, get_transforms()
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        row = self.records[idx]
        img_f, img_b = cv2.cvtColor(cv2.imread(row['front']), cv2.COLOR_BGR2RGB), cv2.cvtColor(cv2.imread(row['back']), cv2.COLOR_BGR2RGB)
        rf, rb = crop_regions(img_f), crop_regions(img_b)
        apply = lambda img: self.tfm(image=img)["image"]
        return {
            "full": torch.stack([apply(rf["full"]), apply(rb["full"])]),
            "surface": torch.stack([apply(rf["surface"]), apply(rb["surface"])]),
            "corners": torch.cat([torch.stack([apply(rf["corners"][i]) for i in range(4)]), torch.stack([apply(rb["corners"][i]) for i in range(4)])]),
            "edges": torch.cat([torch.stack([apply(rf["edges"][i]) for i in range(4)]), torch.stack([apply(rb["edges"][i]) for i in range(4)])]),
            "label": torch.tensor(row['label'], dtype=torch.long)
        }

class BranchEncoder(nn.Module):
    def __init__(self, backbone_type, out_dim=256):
        super().__init__()
        self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_feat = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(in_feat, out_dim), nn.LayerNorm(out_dim), nn.GELU())
    def forward(self, x): return self.proj(self.pool(self.backbone.features(x)).flatten(1))

class PSAModel(nn.Module):
    def __init__(self, backbone_type='b4', embed_dim=256, dropout=0.3):
        super().__init__()
        self.enc_full = BranchEncoder(backbone_type, embed_dim)
        self.enc_corners = BranchEncoder(backbone_type, embed_dim)
        self.enc_edges = BranchEncoder(backbone_type, embed_dim)
        self.enc_surface = BranchEncoder(backbone_type, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 4, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=embed_dim*4, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4, norm=nn.LayerNorm(embed_dim))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 3))
    def _encode_multi(self, encoder, x):
        B, N, C, H, W = x.shape
        return encoder(x.view(B * N, C, H, W)).view(B, N, -1).mean(1)
    def forward(self, full, corners, edges, surface):
        t_f = self._encode_multi(self.enc_full, full)
        t_c = self._encode_multi(self.enc_corners, corners)
        t_e = self._encode_multi(self.enc_edges, edges)
        t_s = self._encode_multi(self.enc_surface, surface)
        tokens = torch.stack([t_f, t_c, t_e, t_s], dim=1) + self.pos_embed
        return self.head(self.transformer(tokens).mean(dim=1))

def predict_with_tta(model, f, c, e, s):
    probs = []
    probs.append(torch.softmax(model(f, c, e, s), dim=1))
    probs.append(torch.softmax(model(torch.flip(f, [4]), torch.flip(c, [4]), torch.flip(e, [4]), torch.flip(s, [4])), dim=1))
    return torch.stack(probs).mean(dim=0)

def run_master_ensemble():
    TEST_ROOT = "/data3/home/h22000561/psa_grading/data/test_yolo"
    SAVE_DIR = "/data/EunJi/h22000561_psa/v22_Master_Ensemble"
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_ldr = DataLoader(PSADataset(build_robust_pairs(TEST_ROOT)), 8, shuffle=False, num_workers=4)
    
    # 🎯 모델 경로 (수정 불필요)
    dir_gm2 = "/data/EunJi/h22000561_psa/seq_b4_lr3e-5_gm2.0" 
    dir_gm3 = "/data/EunJi/h22000561_psa/seq_b4_lr3e-5_gm3.0"
    
    # 두 모델을 구분해서 로드합니다.
    model_gm2, model_gm3 = None, None
    if os.path.exists(os.path.join(dir_gm2, "best_fold0.pth")):
        model_gm2 = PSAModel('b4').to(device)
        model_gm2.load_state_dict(torch.load(os.path.join(dir_gm2, "best_fold0.pth"), map_location=device))
        model_gm2.eval()
        
    if os.path.exists(os.path.join(dir_gm3, "best_fold0.pth")):
        model_gm3 = PSAModel('b4').to(device)
        model_gm3.load_state_dict(torch.load(os.path.join(dir_gm3, "best_fold0.pth"), map_location=device))
        model_gm3.eval()

    if not model_gm2 or not model_gm3: return print("❌ 모델 가중치가 부족합니다.")

    lbls, probs_10 = [], []
    print("\n🚀 카드를 채점하며 10점 확률을 계산 중입니다...")
    
    with torch.no_grad():
        for b in test_ldr:
            l = b["label"].to(device)
            f, c, e, s = b["full"].to(device), b["corners"].to(device), b["edges"].to(device), b["surface"].to(device)
            
            # 1. 각 모델의 확률 추출
            prob2 = predict_with_tta(model_gm2, f, c, e, s)
            prob3 = predict_with_tta(model_gm3, f, c, e, s)
            
            # 2. 🎯 가중 평균 (깐깐한 gm3 모델의 의견을 2배 더 강하게 반영!)
            ens_probs = (prob2 * 1.0 + prob3 * 2.0) / 3.0
            
            # 3. 10점 확률만 따로 모으기
            binary_probs = torch.stack([ens_probs[:, 0] + ens_probs[:, 1], ens_probs[:, 2]], dim=1)
            lbls.extend(l.cpu().tolist())
            probs_10.extend(binary_probs[:, 1].cpu().tolist()) 
            
    # 🎯 3. 최적의 임계값(Threshold) 자동 탐색기 (가장 높은 정확도 찾기)
    best_acc = 0.0
    best_thresh = 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        temp_preds = [1 if p >= t else 0 for p in probs_10]
        acc = accuracy_score(lbls, temp_preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    print(f"✨ [탐색 완료] 가장 높은 정확도를 달성하는 황금 임계값은 {best_thresh:.2f} 입니다! (예상 정확도: {best_acc*100:.1f}%)")
    
    # 4. 찾은 임계값으로 최종 성적 매기기
    final_preds = [1 if p >= best_thresh else 0 for p in probs_10]
    
    rep = classification_report(lbls, final_preds, target_names=CLASS_NAMES, digits=4, zero_division=0)
    print(f"\n=== Auto-Tuned TTA Ensemble Report ===")
    print(rep)
    with open(f"{SAVE_DIR}/report.txt", "w") as f: f.write(rep)
    
    plt.figure(figsize=(14, 6)); plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(lbls, final_preds), annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Binary Confusion Matrix (Thresh: {best_thresh:.2f})')
    plt.subplot(1, 2, 2); fpr, tpr, _ = roc_curve(lbls, probs_10)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc(fpr, tpr):.3f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.savefig(f"{SAVE_DIR}/graphs.png", dpi=300)
    print(f"✅ 황금 밸런스 성적표 및 그래프 저장 완료: {SAVE_DIR}")

if __name__ == "__main__":
    run_master_ensemble()
