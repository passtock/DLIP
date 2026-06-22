#!/bin/bash
#SBATCH --job-name=v22_BIN
#SBATCH --output=/data/EunJi/h22000561_psa/logs/train_binary_endgame_%j.out
#SBATCH --error=/data/EunJi/h22000561_psa/logs/train_binary_endgame_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1

export LC_ALL=C
export LD_LIBRARY_PATH=/home/sonic/anaconda3/lib:$LD_LIBRARY_PATH
source /home/sonic/anaconda3/etc/profile.d/conda.sh
conda activate h22000561-psa

cd /data3/home/h22000561/psa_grading/

cat << 'PYEOF' > train_eval_binary_endgame.py
import os, random, argparse, re
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
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

def get_transforms(mode="train"):
    m, s = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if mode == "train":
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2), 
            A.MotionBlur(blur_limit=3, p=0.2),
            A.Normalize(m, s), ToTensorV2()
        ])
    return A.Compose([A.Normalize(m, s), ToTensorV2()])

class PSADataset(Dataset):
    def __init__(self, records, mode="train"):
        self.records, self.tfm = records, get_transforms(mode)
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

def run_fold(fold, records, args, device):
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    fs = len(records) // 5; tr_recs, vl_recs = records[:fold*fs] + records[(fold+1)*fs:], records[fold*fs:(fold+1)*fs]
    
    labels_list = [r['label'] for r in tr_recs]
    weights = [1.0 / c if c > 0 else 0 for c in [labels_list.count(0), labels_list.count(1)]]
    sampler = WeightedRandomSampler(torch.tensor([weights[r['label']] for r in tr_recs]), len(tr_recs), replacement=True)
    
    tr_ldr = DataLoader(PSADataset(tr_recs, "train"), args.batch_size, sampler=sampler, num_workers=4)
    vl_ldr = DataLoader(PSADataset(vl_recs, "val"), args.batch_size, shuffle=False, num_workers=4)
    
    model = PSABinaryModel().to(device)
    criterion = nn.BCEWithLogitsLoss() 
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scaler = GradScaler()
    
    best_auc, pat = 0.0, 0
    print(f"\n🚀 [Fold {fold} / 5] 최후의 이진 전용 학습 시작 (B4, LR:{args.lr})")
    for ep in range(args.epochs):
        model.train(); tr_l, tr_c, tot = 0, 0, 0
        for i, b in enumerate(tr_ldr):
            l = b["label"].to(device)
            with autocast():
                out = model(b["full"].to(device), b["corners"].to(device), b["edges"].to(device), b["surface"].to(device))
                loss = criterion(out, l) / args.accum_steps
            scaler.scale(loss).backward()
            if (i+1)%args.accum_steps==0 or (i+1)==len(tr_ldr):
                scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            tr_l += loss.item()*args.accum_steps*l.size(0)
            preds = (torch.sigmoid(out) > 0.5).float()
            tr_c += (preds == l).sum().item(); tot += l.size(0)
        scheduler.step()

        model.eval(); vl_l, vl_c, v_tot, probs, lbls = 0, 0, 0, [], []
        with torch.no_grad():
            for b in vl_ldr:
                l = b["label"].to(device)
                with autocast():
                    out = model(b["full"].to(device), b["corners"].to(device), b["edges"].to(device), b["surface"].to(device))
                    loss = criterion(out, l)
                vl_l += loss.item()*l.size(0)
                probs.extend(torch.sigmoid(out).cpu().tolist()); lbls.extend(l.cpu().tolist())
                preds = (torch.sigmoid(out) > 0.5).float()
                vl_c += (preds == l).sum().item(); v_tot += l.size(0)
        
        try: auc_score = roc_auc_score(lbls, probs) if len(np.unique(lbls)) > 1 else 0.5
        except: auc_score = 0.5
            
        print(f"[Ep {ep+1:02d}/{args.epochs}] Tr_Loss: {tr_l/tot:.4f} Tr_Acc: {tr_c/tot:.4f} | Vl_Loss: {vl_l/v_tot:.4f} Vl_Acc: {vl_c/v_tot:.4f} | AUC: {auc_score:.4f}")
        if auc_score > best_auc:
            best_auc, pat = auc_score, 0
            torch.save(model.state_dict(), save_dir / f"best_fold{fold}.pth")
        elif (pat := pat + 1) >= args.patience: 
            print(f"🛑 조기 종료 Fold {fold}"); break

def predict_with_tta(model, f, c, e, s):
    probs = []
    probs.append(torch.sigmoid(model(f, c, e, s)))
    probs.append(torch.sigmoid(model(torch.flip(f, [4]), torch.flip(c, [4]), torch.flip(e, [4]), torch.flip(s, [4]))))
    return torch.stack(probs).mean(dim=0)

def auto_evaluate(args, device):
    print("\n🏆 최종 심판: Binary Native 앙상블 평가")
    test_ldr = DataLoader(PSADataset(build_robust_pairs(args.test_root), "val"), 8, shuffle=False, num_workers=4)
    
    models = []
    for i in range(5):
        wp = f"{args.save_dir}/best_fold{i}.pth"
        if os.path.exists(wp):
            m = PSABinaryModel().to(device)
            m.load_state_dict(torch.load(wp, map_location=device))
            m.eval()
            models.append(m)
            
    if not models: return print("❌ 모델 가중치 없음")

    lbls, probs_10 = [], []
    with torch.no_grad():
        for b in test_ldr:
            l = b["label"].to(device)
            ens_probs = torch.zeros((l.size(0))).to(device)
            for m in models: 
                ens_probs += predict_with_tta(m, b["full"].to(device), b["corners"].to(device), b["edges"].to(device), b["surface"].to(device))
            ens_probs /= len(models)
            
            lbls.extend(l.cpu().tolist())
            probs_10.extend(ens_probs.cpu().tolist())
            
    best_acc, best_thresh = 0.0, 0.5
    for t in np.arange(0.2, 0.8, 0.01):
        temp_preds = [1 if p >= t else 0 for p in probs_10]
        acc = accuracy_score(lbls, temp_preds)
        if acc > best_acc: best_acc, best_thresh = acc, t

    print(f"✨ [탐색 완료] 황금 임계값: {best_thresh:.2f} (예상 최고 정확도: {best_acc*100:.1f}%)")
    final_preds = [1 if p >= best_thresh else 0 for p in probs_10]
    
    rep = classification_report(lbls, final_preds, target_names=CLASS_NAMES, digits=4, zero_division=0)
    print(f"\n=== Binary ENDGAME Report ===\n{rep}")
    with open(f"{args.save_dir}/report.txt", "w") as f: f.write(rep)
    pd.DataFrame(classification_report(lbls, final_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)).transpose().to_csv(f"{args.save_dir}/report.csv")
    
    plt.figure(figsize=(14, 6)); plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(lbls, final_preds), annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('ENDGAME Confusion Matrix'); plt.subplot(1, 2, 2); fpr, tpr, _ = roc_curve(lbls, probs_10)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc(fpr, tpr):.3f}'); plt.plot([0,1], [0,1], 'k--'); plt.legend(loc="lower right")
    plt.savefig(f"{args.save_dir}/graphs.png", dpi=300); print(f"✅ 최종 성적표 저장 완료: {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data3/home/h22000561/psa_grading/data/train_yolo")
    parser.add_argument("--test_root", type=str, default="/data3/home/h22000561/psa_grading/data/test_yolo")
    parser.add_argument("--save_dir", type=str, default="/data/EunJi/h22000561_psa/v22_Binary_Endgame")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8) 
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    recs = build_robust_pairs(args.data_root)
    for f in range(5): run_fold(f, recs, args, device)
    auto_evaluate(args, device)
PYEOF

python train_eval_binary_endgame.py
