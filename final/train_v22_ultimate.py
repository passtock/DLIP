import os, random, argparse, re
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from sklearn.metrics import roc_auc_score, classification_report
import albumentations as A
from albumentations.pytorch import ToTensorV2

CFG = {'full': 512, 'corner': 256, 'edge': 128, 'surface': 384}
CLASS_NAMES = ["PSA 8", "PSA 9", "PSA 10"]

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
    print(f"✅ 완벽한 앞/뒷면 페어 데이터 로드 완료: 총 {len(records)}세트")
    return records

def crop_regions(img: np.ndarray) -> dict:
    H, W = img.shape[:2]
    full = cv2.resize(img, (CFG['full'], CFG['full']))
    cs = 96 
    corners = np.stack([
        cv2.resize(img[0:cs, 0:cs], (CFG['corner'], CFG['corner'])),
        cv2.resize(img[0:cs, W-cs:W], (CFG['corner'], CFG['corner'])),
        cv2.resize(img[H-cs:H, 0:cs], (CFG['corner'], CFG['corner'])),
        cv2.resize(img[H-cs:H, W-cs:W], (CFG['corner'], CFG['corner']))
    ], axis=0)
    es = 32
    edges = np.stack([
        cv2.resize(img[0:es, :], (CFG['edge'], CFG['edge'])),
        cv2.resize(img[H-es:H, :], (CFG['edge'], CFG['edge'])),
        cv2.resize(img[:, 0:es], (CFG['edge'], CFG['edge'])),
        cv2.resize(img[:, W-es:W], (CFG['edge'], CFG['edge']))
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
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    return A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])

def apply_transform(tfm, img_np: np.ndarray) -> torch.Tensor:
    return tfm(image=img_np)["image"]

class PSADataset_v22(Dataset):
    def __init__(self, records: list, mode: str = "train"):
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
        self.backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(in_features, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        
        # 백본 동결!
        for param in self.backbone.parameters():
            param.requires_grad = False
            
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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scaler = GradScaler()
    best_auc, patience_cnt = 0.0, 0
    print(f"\n🚀 [Fold {fold}] 백본 동결 가동 ➔ 과적합 차단 모드 시작")
    
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
            if len(np.unique(all_labels)) > 1: val_auc = roc_auc_score(all_labels, all_probs)
            else: val_auc = 0.5
        except: val_auc = 0.5

        print(f"[Fold {fold} | Ep {epoch+1:02d}/{args.epochs}] Tr_Loss: {tr_loss/total:.4f} Tr_Acc: {tr_correct/total:.4f} | Vl_Loss: {vl_loss/v_total:.4f} Vl_Acc: {vl_correct/v_total:.4f} | AUC(10점): {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc, patience_cnt = val_auc, 0
            torch.save(model.state_dict(), save_dir / f"best_fold{fold}.pth")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience: print(f"🛑 [Early stopping] Fold {fold}"); break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data3/home/h22000561/psa_grading/data/train_yolo")
    parser.add_argument("--save_dir", type=str, default="/data/EunJi/h22000561_psa/v22_ultimate")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)   
    parser.add_argument("--accum_steps", type=int, default=8)  
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = build_robust_pairs(args.data_root)
    
    # 여기서 5개 폴드가 다 돌아갑니다!
    for f in range(5):
        run_fold(f, records, args, device)
        print(f"✅ Fold {f} 완료! 다음 Fold로 넘어갑니다...\n" + "="*50)
