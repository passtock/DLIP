import os, re, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import cv2
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================
# 1. 환경 설정
# ==========================================
TEST_ROOT = '/data3/home/h22000561/psa_grading/data/test_yolo'
RES_DIR = '/data/EunJi/h22000561_psa/v22_slow_learn'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CFG = {'full': 512, 'corner': 256, 'edge': 128, 'surface': 384}
CLASS_NAMES = ["PSA 8", "PSA 9", "PSA 10"]

# ==========================================
# 2. 데이터셋 (엄격한 153세트 룰 적용)
# ==========================================
def build_robust_pairs(data_root):
    records = []
    for grade in [8, 9, 10]:
        label = 0 if grade == 8 else (1 if grade == 9 else 2)
        folder = next((Path(data_root) / f for f in [f'PSA{grade}', f'psa_{grade}', f'psa{grade}', f'PSA_{grade}'] if (Path(data_root) / f).exists()), None)
        if not folder: continue
        cert_dict = defaultdict(dict)
        for img_path in folder.glob('*.*'):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}: continue
            match = re.search(r'(cert\d+)', img_path.stem.lower())
            if match:
                cid = match.group(1)
                cert_dict[cid]['front' if 'front' in img_path.stem.lower() else 'back'] = str(img_path)
        for sides in cert_dict.values():
            if 'front' in sides and 'back' in sides:
                records.append({'front': sides['front'], 'back': sides['back'], 'label': label})
    return records

def crop_regions(img: np.ndarray) -> dict:
    H, W = img.shape[:2]
    full = cv2.resize(img, (CFG['full'], CFG['full']))
    cs = 96 
    corners = np.stack([
        cv2.resize(img[0:cs, 0:cs], (CFG['corner'], CFG['corner'])), cv2.resize(img[0:cs, W-cs:W], (CFG['corner'], CFG['corner'])),
        cv2.resize(img[H-cs:H, 0:cs], (CFG['corner'], CFG['corner'])), cv2.resize(img[H-cs:H, W-cs:W], (CFG['corner'], CFG['corner']))
    ], axis=0)
    es = 32
    edges = np.stack([
        cv2.resize(img[0:es, :], (CFG['edge'], CFG['edge'])), cv2.resize(img[H-es:H, :], (CFG['edge'], CFG['edge'])),
        cv2.resize(img[:, 0:es], (CFG['edge'], CFG['edge'])), cv2.resize(img[:, W-es:W], (CFG['edge'], CFG['edge']))
    ], axis=0)
    cy, cx, r = H // 2, W // 2, 128
    surface = cv2.resize(img[cy-r:cy+r, cx-r:cx+r], (CFG['surface'], CFG['surface']))
    return {"full": full, "corners": corners, "edges": edges, "surface": surface}

def get_transforms():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])

def apply_transform(tfm, img_np: np.ndarray) -> torch.Tensor:
    return tfm(image=img_np)["image"]

class PSADataset_v22(Dataset):
    def __init__(self, records: list):
        self.records = records
        self.transform = get_transforms()
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        row = self.records[idx]
        img_f = cv2.cvtColor(cv2.imread(row['front']), cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(cv2.imread(row['back']), cv2.COLOR_BGR2RGB)
        rf, rb = crop_regions(img_f), crop_regions(img_b)
        t = self.transform
        f_tensor = torch.stack([apply_transform(t, rf["full"]), apply_transform(t, rb["full"])])
        s_tensor = torch.stack([apply_transform(t, rf["surface"]), apply_transform(t, rb["surface"])])
        c_tensor = torch.cat([torch.stack([apply_transform(t, rf["corners"][i]) for i in range(4)]), torch.stack([apply_transform(t, rb["corners"][i]) for i in range(4)])], dim=0)
        e_tensor = torch.cat([torch.stack([apply_transform(t, rf["edges"][i]) for i in range(4)]), torch.stack([apply_transform(t, rb["edges"][i]) for i in range(4)])], dim=0)
        return {"full": f_tensor, "corners": c_tensor, "edges": e_tensor, "surface": s_tensor, "label": torch.tensor(row['label'], dtype=torch.long)}

# ==========================================
# 3. 모델 아키텍처 (B2)
# ==========================================
class BranchEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(in_features, out_dim), nn.LayerNorm(out_dim), nn.GELU())
    def forward(self, x): return self.proj(self.pool(self.backbone.features(x)).flatten(1))

class PSAUltimateModel(nn.Module):
    def __init__(self, embed_dim=256, n_heads=8, n_layers=4, dropout=0.3):
        super().__init__()
        self.enc_full = BranchEncoder(embed_dim)
        self.enc_corners = BranchEncoder(embed_dim)
        self.enc_edges = BranchEncoder(embed_dim)
        self.enc_surface = BranchEncoder(embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 4, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*4, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(embed_dim))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 3))

    def _encode_multi(self, encoder, x):
        B, N, C, H, W = x.shape
        return encoder(x.view(B * N, C, H, W)).view(B, N, -1).mean(1)

    def forward(self, full, corners, edges, surface):
        tokens = torch.stack([self._encode_multi(self.enc_full, full), self._encode_multi(self.enc_corners, corners), self._encode_multi(self.enc_edges, edges), self._encode_multi(self.enc_surface, surface)], dim=1) + self.pos_embed
        return self.head(self.transformer(tokens).mean(dim=1))

# ==========================================
# 4. 5-Fold 앙상블 테스트 평가
# ==========================================
def evaluate_ensemble():
    print(f"\n🚀 [v22_slow_learn] 5-Fold 앙상블 테스트 평가 시작...")
    records = build_robust_pairs(TEST_ROOT)
    test_ldr = DataLoader(PSADataset_v22(records), batch_size=8, shuffle=False, num_workers=4)

    # 5개의 모델 모두 로드
    models = []
    for i in range(5):
        weight_path = os.path.join(RES_DIR, f'best_fold{i}.pth')
        if os.path.exists(weight_path):
            model = PSAUltimateModel().to(DEVICE)
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
            model.eval()
            models.append(model)
            print(f"✅ Fold {i} 가중치 로드 완료")
    
    if not models:
        print("❌ 저장된 가중치가 없습니다!")
        return

    all_preds, all_labels, all_probs_10 = [], [], []

    print("📊 채점 중... (5개의 AI가 다수결로 판정합니다)")
    with torch.no_grad():
        for batch in test_ldr:
            f, c, e, s, l = batch["full"].to(DEVICE), batch["corners"].to(DEVICE), batch["edges"].to(DEVICE), batch["surface"].to(DEVICE), batch["label"].to(DEVICE)
            
            # 5개 모델의 예측 확률 평균내기 (Ensemble)
            ensemble_probs = torch.zeros((f.size(0), 3)).to(DEVICE)
            for m in models:
                out = m(f, c, e, s)
                ensemble_probs += torch.softmax(out, dim=1)
            ensemble_probs /= len(models)
            
            all_preds.extend(ensemble_probs.argmax(dim=1).cpu().tolist())
            all_labels.extend(l.cpu().tolist())
            all_probs_10.extend(ensemble_probs[:, 2].cpu().tolist())

    # 1. TXT / CSV 리포트 생성
    report_text = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)
    print("\n=== v22_slow_learn Ensemble Test Report ===")
    print(report_text)
    
    with open(os.path.join(RES_DIR, 'test_summary_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== v22_slow_learn 5-Fold Ensemble Test Report ===\n")
        f.write(report_text)
    
    report_dict = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv(os.path.join(RES_DIR, 'test_summary_report.csv'))

    # 2. 그래프 시각화
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('v22_slow Ensemble Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('Actual')

    plt.subplot(1, 2, 2)
    binary_labels = [1 if lbl == 2 else 0 for lbl in all_labels]
    fpr, tpr, _ = roc_curve(binary_labels, all_probs_10)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'10-Point ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('v22_slow Ensemble ROC Curve (PSA 10 vs Rest)')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    graph_path = os.path.join(RES_DIR, 'test_graphs.png')
    plt.tight_layout()
    plt.savefig(graph_path, dpi=300)
    print(f"✅ TXT 리포트 및 그래프 저장 완료: {RES_DIR}")

if __name__ == '__main__':
    evaluate_ensemble()
