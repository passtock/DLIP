import os, re, torch, cv2
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from collections import defaultdict
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm  # 🎯 진행률 바 라이브러리 추가!

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
            "label": torch.tensor(row['label'], dtype=torch.float32)
        }

class BranchEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
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

def predict_with_tta(model, f, c, e, s):
    probs = []
    probs.append(torch.sigmoid(model(f, c, e, s)))
    probs.append(torch.sigmoid(model(torch.flip(f, [4]), torch.flip(c, [4]), torch.flip(e, [4]), torch.flip(s, [4]))))
    return torch.stack(probs).mean(dim=0)

def run_train_evaluation():
    TRAIN_ROOT = "/data3/home/h22000561/psa_grading/data/train_yolo"
    LOAD_DIR = "/data/EunJi/h22000561_psa/v22_Binary_Endgame"
    
    os.makedirs(LOAD_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n📚 교과서 복습 심사: 엔드게임 모델의 '학습 데이터(Train Set)' 구체적 채점을 시작합니다!")
    train_ldr = DataLoader(PSADataset(build_robust_pairs(TRAIN_ROOT)), 8, shuffle=False, num_workers=4)
    
    models = []
    for i in range(5):
        wp = os.path.join(LOAD_DIR, f"best_fold{i}.pth")
        if os.path.exists(wp):
            m = PSABinaryModel().to(device)
            m.load_state_dict(torch.load(wp, map_location=device))
            m.eval()
            models.append(m)
            
    if not models: return print("❌ 모델 가중치를 찾을 수 없습니다. 경로를 확인해주세요.")

    lbls, probs_10 = [], []
    
    print("\n⏳ 5-Fold x TTA 앙상블 채점 중입니다. 잠시만 기다려주세요...")
    with torch.no_grad():
        # 🎯 tqdm으로 감싸서 진행률 바 출력!
        for b in tqdm(train_ldr, desc="[채점 진행률]", unit="batch"):
            l = b["label"].to(device)
            ens_probs = torch.zeros((l.size(0))).to(device)
            
            for m in models: 
                ens_probs += predict_with_tta(m, b["full"].to(device), b["corners"].to(device), b["edges"].to(device), b["surface"].to(device))
            ens_probs /= len(models)
            
            lbls.extend(l.cpu().tolist())
            probs_10.extend(ens_probs.cpu().tolist())
            
    best_thresh = 0.59
    final_preds = [1 if p >= best_thresh else 0 for p in probs_10]
    
    rep = classification_report(lbls, final_preds, target_names=CLASS_NAMES, digits=4, zero_division=0)
    print(f"\n\n=== 🏅 ENDGAME 학습 데이터(TRAIN SET) Final Report 🏅 ===")
    print(f"(적용된 황금 임계값: {best_thresh})\n")
    print(rep)
    
    with open(f"{LOAD_DIR}/TRAIN_SET_report.txt", "w") as f: f.write(rep)
    
    plt.figure(figsize=(14, 6)); plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(lbls, final_preds), annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'TRAIN SET Matrix (Thresh: {best_thresh:.2f})'); plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.subplot(1, 2, 2); fpr, tpr, _ = roc_curve(lbls, probs_10)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc(fpr, tpr):.3f}')
    plt.plot([0,1], [0,1], 'k--'); plt.legend(loc="lower right")
    plt.title('TRAIN SET ROC Curve')
    plt.savefig(f"{LOAD_DIR}/TRAIN_SET_graphs.png", dpi=300)
    
    print(f"✅ 학습 데이터 구체적 성적표 및 그래프 저장 완료: {LOAD_DIR}")

if __name__ == "__main__":
    run_train_evaluation()
