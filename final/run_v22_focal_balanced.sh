#!/bin/bash
#SBATCH --job-name=v22_F_Bal
#SBATCH --output=/data/EunJi/h22000561_psa/logs/train_v22_focal_bal_%j.out
#SBATCH --error=/data/EunJi/h22000561_psa/logs/train_v22_focal_bal_%j.err
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1

mkdir -p /data/EunJi/h22000561_psa/logs
export LD_LIBRARY_PATH=/home/sonic/anaconda3/lib:$LD_LIBRARY_PATH
source /home/sonic/anaconda3/etc/profile.d/conda.sh
conda activate h22000561-psa

cd /data3/home/h22000561/psa_grading/

cat << 'PYEOF' > train_eval_v22_focal_bal.py
import os, random, argparse, re
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2

CFG = {'full': 512, 'corner': 256, 'edge': 128, 'surface': 384}
CLASS_NAMES = ["PSA 8", "PSA 9", "PSA 10"]

# ==========================================
# 🎯 Balanced Focal Loss (편애 금지, 어려운 문제만 집중)
# ==========================================
class FocalLoss(nn.Module):
    # Alpha를 모두 1.0으로 주어 특정 클래스를 무서워하지 않게 만듭니다!
    def __init__(self, alpha=[1.0, 1.0, 1.0], gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[targets] * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

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

def get_transforms(mode="train"):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if mode == "train":
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.3),
            A.Normalize(mean=mean, std=std), ToTensorV2()
        ])
    return A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])

def apply_transform(tfm, img_np: np.ndarray) -> torch.Tensor:
    return tfm(image=img_np)["image"]

class PSADataset_v22(Dataset):
    def __init__(self, records: list, mode="train"):
        self.records = records
        self.transform = get_transforms(mode)
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

class BranchEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
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

def get_fold_split(records, n_folds=5, val_fold=0):
    fold_size = len(records) // n_folds
    val_start, val_end = val_fold * fold_size, (val_fold + 1) * fold_size
    return records[:val_start] + records[val_end:], records[val_start:val_end]

def run_fold(fold, records, args, device):
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    train_recs, val_recs = get_fold_split(records, val_fold=fold)
    labels = [r['label'] for r in train_recs]
    class_count = [labels.count(i) for i in range(3)]
    weights = [1.0 / c if c > 0 else 0 for c in class_count]
    sampler = WeightedRandomSampler(weights=torch.tensor([weights[l] for l in labels]), num_samples=len(labels), replacement=True)
    
    train_loader = DataLoader(PSADataset_v22(train_recs, "train"), batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    val_loader = DataLoader(PSADataset_v22(val_recs, "val"), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = PSAUltimateModel().to(device)
    criterion = FocalLoss() # 🎯 밸런스 패치 완료!
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scaler = GradScaler()
    best_auc, patience_cnt = 0.0, 0
    
    print(f"\n🚀 [Fold {fold}] Balanced Focal + B4 훈련 시작")
    for epoch in range(args.epochs):
        model.train(); tr_loss, tr_correct, total = 0, 0, 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            labels_t = batch["label"].to(device)
            with autocast():
                logits = model(batch["full"].to(device), batch["corners"].to(device), batch["edges"].to(device), batch["surface"].to(device))
                loss = criterion(logits, labels_t) / args.accum_steps
            scaler.scale(loss).backward()
            if (i + 1) % args.accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            tr_loss += loss.item() * args.accum_steps * labels_t.size(0)
            tr_correct += (logits.argmax(1) == labels_t).sum().item(); total += labels_t.size(0)
        scheduler.step()

        model.eval(); vl_loss, vl_correct, v_total = 0, 0, 0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                labels_t = batch["label"].to(device)
                with autocast():
                    logits = model(batch["full"].to(device), batch["corners"].to(device), batch["edges"].to(device), batch["surface"].to(device))
                    loss = criterion(logits, labels_t)
                vl_loss += loss.item() * labels_t.size(0)
                vl_correct += (logits.argmax(1) == labels_t).sum().item(); v_total += labels_t.size(0)
                all_probs.extend(torch.softmax(logits, dim=-1)[:, 2].cpu().tolist())
                all_labels.extend((labels_t == 2).cpu().numpy().astype(int))

        try:
            val_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
        except: val_auc = 0.5

        print(f"[Fold {fold} | Ep {epoch+1:02d}/{args.epochs}] Tr_Loss: {tr_loss/total:.4f} Tr_Acc: {tr_correct/total:.4f} | Vl_Loss: {vl_loss/v_total:.4f} Vl_Acc: {vl_correct/v_total:.4f} | AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc, patience_cnt = val_auc, 0
            torch.save(model.state_dict(), save_dir / f"best_fold{fold}.pth")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience: print(f"🛑 조기 종료 Fold {fold}"); break

# ==========================================
# 🎯 [자동 평가] 
# ==========================================
def auto_evaluate(save_dir, test_root, device):
    print("\n" + "="*50)
    print("🏆 모든 Fold 학습 완료! 즉시 5-Fold 앙상블 자동 평가를 시작합니다.")
    print("="*50 + "\n")
    
    records = build_robust_pairs(test_root)
    test_ldr = DataLoader(PSADataset_v22(records, "val"), batch_size=8, shuffle=False, num_workers=4)

    models = []
    for i in range(5):
        weight_path = os.path.join(save_dir, f'best_fold{i}.pth')
        if os.path.exists(weight_path):
            model = PSAUltimateModel().to(device)
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model.eval()
            models.append(model)
            
    if not models:
        print("❌ 저장된 가중치가 없어 평가를 종료합니다.")
        return

    all_preds, all_labels, all_probs_10 = [], [], []
    with torch.no_grad():
        for batch in test_ldr:
            f, c, e, s, l = batch["full"].to(device), batch["corners"].to(device), batch["edges"].to(device), batch["surface"].to(device), batch["label"].to(device)
            ensemble_probs = torch.zeros((f.size(0), 3)).to(device)
            for m in models:
                ensemble_probs += torch.softmax(m(f, c, e, s), dim=1)
            ensemble_probs /= len(models)
            all_preds.extend(ensemble_probs.argmax(dim=1).cpu().tolist())
            all_labels.extend(l.cpu().tolist())
            all_probs_10.extend(ensemble_probs[:, 2].cpu().tolist())

    report_text = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0)
    print("\n=== All-in-One: Balanced Focal Ensemble Test Report ===")
    print(report_text)
    
    with open(os.path.join(save_dir, 'test_summary_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    pd.DataFrame(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)).transpose().to_csv(os.path.join(save_dir, 'test_summary_report.csv'))

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Ensemble Confusion Matrix')
    
    plt.subplot(1, 2, 2)
    binary_labels = [1 if lbl == 2 else 0 for lbl in all_labels]
    fpr, tpr, _ = roc_curve(binary_labels, all_probs_10)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc(fpr, tpr):.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC Curve (PSA 10 vs Rest)')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_graphs.png'), dpi=300)
    print(f"✅ 성적표 및 그래프 자동 저장 완료: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data3/home/h22000561/psa_grading/data/train_yolo")
    parser.add_argument("--test_root", type=str, default="/data3/home/h22000561/psa_grading/data/test_yolo")
    parser.add_argument("--save_dir", type=str, default="/data/EunJi/h22000561_psa/v22_focal_bal")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)   
    parser.add_argument("--accum_steps", type=int, default=8)  
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    records = build_robust_pairs(args.data_root)
    for f in range(5):
        run_fold(f, records, args, device)
        
    auto_evaluate(args.save_dir, args.test_root, device)
PYEOF

python train_eval_v22_focal_bal.py
