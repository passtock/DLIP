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
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2

CFG = {'full': 512, 'corner': 256, 'edge': 128, 'surface': 384}
CLASS_NAMES = ["PSA 8", "PSA 9", "PSA 10"]

class FocalLoss(nn.Module):
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
    if mode == "train": return A.Compose([A.RandomBrightnessContrast(0.15, 0.15, p=0.5), A.HueSaturationValue(5, 10, 10, p=0.3), A.Normalize(m, s), ToTensorV2()])
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
            "label": torch.tensor(row['label'], dtype=torch.long)
        }

class BranchEncoder(nn.Module):
    def __init__(self, backbone_type, out_dim=256):
        super().__init__()
        if backbone_type == 'b4':
            self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        else:
            self.backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
            
        in_feat = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(in_feat, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        
    def forward(self, x): 
        return self.proj(self.pool(self.backbone.features(x)).flatten(1))

class PSAModel(nn.Module):
    def __init__(self, backbone_type, embed_dim=256, dropout=0.3):
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
        out = encoder(x.view(B * N, C, H, W))
        return out.view(B, N, -1).mean(1)

    def forward(self, full, corners, edges, surface):
        # 🎯 버그 수정 (압축 해제): 데이터 타입을 명확하게 맞추어 Transformer 충돌 방지
        t_f = self._encode_multi(self.enc_full, full)
        t_c = self._encode_multi(self.enc_corners, corners)
        t_e = self._encode_multi(self.enc_edges, edges)
        t_s = self._encode_multi(self.enc_surface, surface)
        
        tokens = torch.stack([t_f, t_c, t_e, t_s], dim=1) + self.pos_embed
        
        # Transformer 연산 수행
        trans_out = self.transformer(tokens)
        
        # 최종 분류 
        return self.head(trans_out.mean(dim=1))

def run_fold(fold, records, args, device):
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    fs = len(records) // 5; tr_recs, vl_recs = records[:fold*fs] + records[(fold+1)*fs:], records[fold*fs:(fold+1)*fs]
    
    labels_list = [r['label'] for r in tr_recs]
    weights = [1.0 / c if c > 0 else 0 for c in [labels_list.count(i) for i in range(3)]]
    sampler = WeightedRandomSampler(torch.tensor([weights[r['label']] for r in tr_recs]), len(tr_recs), replacement=True)
    
    tr_ldr = DataLoader(PSADataset(tr_recs, "train"), args.batch_size, sampler=sampler, num_workers=4)
    vl_ldr = DataLoader(PSADataset(vl_recs, "val"), args.batch_size, shuffle=False, num_workers=4)
    
    model = PSAModel(args.backbone).to(device)
    criterion = FocalLoss(gamma=args.gamma)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scaler = GradScaler()
    
    best_auc, pat = 0.0, 0
    print(f"\n🚀 [Fold {fold}] Backbone: {args.backbone.upper()} | LR: {args.lr} | Gamma: {args.gamma}")
    for ep in range(args.epochs):
        model.train(); tr_l, tr_c, tot = 0, 0, 0
        for i, b in enumerate(tr_ldr):
            l = b["label"].to(device)
            # 🎯 확실한 언패킹: 딕셔너리에서 직접 꺼내어 전달 (autocast 충돌 방지)
            with autocast():
                out = model(b["full"].to(device), b["corners"].to(device), b["edges"].to(device), b["surface"].to(device))
                loss = criterion(out, l) / args.accum_steps
            scaler.scale(loss).backward()
            if (i+1)%args.accum_steps==0 or (i+1)==len(tr_ldr):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            tr_l += loss.item()*args.accum_steps*l.size(0); tr_c += (out.argmax(1)==l).sum().item(); tot += l.size(0)
        scheduler.step()

        model.eval(); vl_l, vl_c, v_tot, probs, lbls = 0, 0, 0, [], []
        with torch.no_grad():
            for b in vl_ldr:
                l = b["label"].to(device)
                with autocast():
                    out = model(b["full"].to(device), b["corners"].to(device), b["edges"].to(device), b["surface"].to(device))
                    loss = criterion(out, l)
                vl_l += loss.item()*l.size(0); vl_c += (out.argmax(1)==l).sum().item(); v_tot += l.size(0)
                probs.extend(torch.softmax(out, -1)[:,2].cpu().tolist()); lbls.extend((l==2).cpu().tolist())
        
        try: auc_score = roc_auc_score(lbls, probs) if len(np.unique(lbls)) > 1 else 0.5
        except: auc_score = 0.5
            
        print(f"[Ep {ep+1:02d}] Tr_Loss: {tr_l/tot:.4f} Tr_Acc: {tr_c/tot:.4f} | Vl_Loss: {vl_l/v_tot:.4f} Vl_Acc: {vl_c/v_tot:.4f} | AUC: {auc_score:.4f}")
        if auc_score > best_auc:
            best_auc, pat = auc_score, 0
            torch.save(model.state_dict(), save_dir / f"best_fold{fold}.pth")
        elif (pat := pat + 1) >= 5:
            print(f"🛑 Early stopping Fold {fold}"); break

def auto_evaluate(args, device):
    print("\n🏆 자동 평가 시작!")
    test_ldr = DataLoader(PSADataset(build_robust_pairs(args.test_root), "val"), 8, shuffle=False, num_workers=4)
    
    models = []
    for i in range(5):
        wp = f"{args.save_dir}/best_fold{i}.pth"
        if os.path.exists(wp):
            m = PSAModel(args.backbone).to(device)
            m.load_state_dict(torch.load(wp, map_location=device))
            m.eval()
            models.append(m)
            
    if not models: return print("❌ 모델 가중치 없음")

    preds, lbls, probs = [], [], []
    with torch.no_grad():
        for b in test_ldr:
            l = b["label"].to(device)
            ens_probs = torch.zeros((l.size(0), 3)).to(device)
            for m in models:
                ens_probs += torch.softmax(m(b["full"].to(device), b["corners"].to(device), b["edges"].to(device), b["surface"].to(device)), dim=1)
            ens_probs /= len(models)
            preds.extend(ens_probs.argmax(1).cpu().tolist())
            lbls.extend(l.cpu().tolist())
            probs.extend(ens_probs[:,2].cpu().tolist())
            
    rep = classification_report(lbls, preds, target_names=CLASS_NAMES, digits=4, zero_division=0)
    print(f"\n=== {args.exp_name} Report ===\n{rep}")
    with open(f"{args.save_dir}/report.txt", "w") as f: f.write(rep)
    pd.DataFrame(classification_report(lbls, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)).transpose().to_csv(f"{args.save_dir}/report.csv")
    
    plt.figure(figsize=(14, 6)); plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(lbls, preds), annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.subplot(1, 2, 2); fpr, tpr, _ = roc_curve([1 if l==2 else 0 for l in lbls], probs)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc(fpr, tpr):.3f}'); plt.plot([0,1], [0,1], 'k--'); plt.legend(loc="lower right")
    plt.title('ROC Curve')
    plt.savefig(f"{args.save_dir}/graphs.png", dpi=300); print(f"✅ 저장 완료: {args.save_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=str, default="b2")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--data_root", type=str, default="/data3/home/h22000561/psa_grading/data/train_yolo")
    p.add_argument("--test_root", type=str, default="/data3/home/h22000561/psa_grading/data/test_yolo")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accum_steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args(); args.save_dir = f"/data/EunJi/h22000561_psa/{args.exp_name}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recs = build_robust_pairs(args.data_root)
    for f in range(5): run_fold(f, recs, args, device)
    auto_evaluate(args, device)
