import os, random, argparse, time
from pathlib import Path
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

# ════════════════════════════════════════════════════════
# 0. 리눅스 서버 맞춤형 경로 및 설정
# ════════════════════════════════════════════════════════
# YOLO 누끼 폴더명에 맞춤
GRADE_TO_LABEL = {"PSA8": 0, "PSA9": 1, "PSA10": 2, "psa8": 0, "psa9": 1, "psa10": 2}
EXTS           = {".jpg", ".jpeg", ".png", ".webp"}
CARD_H, CARD_W = 712, 512

def crop_regions(img: np.ndarray) -> dict:
    H, W = img.shape[:2]
    full = cv2.resize(img, (224, 224))
    cs = 96
    corners = np.stack([
        cv2.resize(img[0:cs, 0:cs], (96, 96)), cv2.resize(img[0:cs, W-cs:W], (96, 96)),
        cv2.resize(img[H-cs:H, 0:cs], (96, 96)), cv2.resize(img[H-cs:H, W-cs:W], (96, 96))
    ], axis=0)
    es = 32
    edges = np.stack([
        cv2.resize(img[0:es, :], (224, 64)), cv2.resize(img[H-es:H, :], (224, 64)),
        cv2.resize(img[:, 0:es], (224, 64)), cv2.resize(img[:, W-es:W], (224, 64))
    ], axis=0)
    cy, cx, r = H // 2, W // 2, 96
    surface = cv2.resize(img[cy-r:cy+r, cx-r:cx+r], (192, 192))
    return {"full": full, "corners": corners, "edges": edges, "surface": surface}

def get_transforms(mode: str):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if mode == "train":
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4),
            A.GaussNoise(var_limit=(5, 20), p=0.3),
            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.3),
            # 센터링 유지를 위해 회전/뒤집기 금지
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    return A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])

def apply_transform(tfm, img_np: np.ndarray) -> torch.Tensor:
    return tfm(image=img_np)["image"]

class PSADataset(Dataset):
    def __init__(self, file_list: list, mode: str = "train"):
        self.file_list, self.transform = file_list, get_transforms(mode)
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (CARD_W, CARD_H))
        r, t = crop_regions(img), self.transform
        return {
            "full": apply_transform(t, r["full"]),
            "surface": apply_transform(t, r["surface"]),
            "corners": torch.stack([apply_transform(t, r["corners"][i]) for i in range(4)]),
            "edges": torch.stack([apply_transform(t, r["edges"][i]) for i in range(4)]),
            "label": torch.tensor(label, dtype=torch.long),
        }

def load_file_list(data_root: str) -> list:
    file_list = []
    for grade_folder in Path(data_root).iterdir():
        if grade_folder.is_dir() and grade_folder.name in GRADE_TO_LABEL:
            label = GRADE_TO_LABEL[grade_folder.name]
            for f in grade_folder.glob('*.*'):
                if f.suffix.lower() in EXTS: file_list.append((str(f), label))
    random.shuffle(file_list)
    print(f"전체 로드된 이미지: {len(file_list)}장")
    return file_list

def get_fold_split(file_list: list, n_folds: int = 5, val_fold: int = 0):
    fold_size = len(file_list) // n_folds
    val_start, val_end = val_fold * fold_size, (val_fold + 1) * fold_size
    return file_list[:val_start] + file_list[val_end:], file_list[val_start:val_end]

def build_dataloaders(file_list, val_fold, batch_size, num_workers):
    train_list, val_list = get_fold_split(file_list, val_fold=val_fold)
    train_ds, val_ds = PSADataset(train_list, "train"), PSADataset(val_list, "val")
    
    labels = [lbl for _, lbl in train_list]
    class_count = [labels.count(i) for i in range(3)]
    class_w = [1.0 / c if c > 0 else 0 for c in class_count]
    sampler = WeightedRandomSampler(weights=torch.tensor([class_w[l] for l in labels]), num_samples=len(labels), replacement=True)

    print(f"[Fold {val_fold}] Train: {len(train_ds)}장 | Val: {len(val_ds)}장 | 클래스 분포(8/9/10): {class_count}")
    return DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=True), DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

class BranchEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(in_features, out_dim), nn.LayerNorm(out_dim), nn.GELU())
    def forward(self, x): return self.proj(self.pool(self.backbone.features(x)).flatten(1))

class PSAGradingModel(nn.Module):
    def __init__(self, embed_dim=256, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.enc_full, self.enc_corners, self.enc_edges, self.enc_surface = BranchEncoder(embed_dim), BranchEncoder(embed_dim), BranchEncoder(embed_dim), BranchEncoder(embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 4, embed_dim) * 0.02)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*4, dropout=dropout, activation="gelu", batch_first=True, norm_first=True), num_layers=n_layers, norm=nn.LayerNorm(embed_dim))
        self.head = nn.Sequential(nn.Linear(embed_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 3)) # 3 Classes (8, 9, 10)
    
    def _encode_multi(self, encoder, x):
        B, N, C, H, W = x.shape
        return encoder(x.view(B * N, C, H, W)).view(B, N, -1).mean(1)
        
    def forward(self, full, corners, edges, surface):
        tokens = torch.stack([self.enc_full(full), self._encode_multi(self.enc_corners, corners), self._encode_multi(self.enc_edges, edges), self.enc_surface(surface)], dim=1) + self.pos_embed
        return self.head(self.transformer(tokens).mean(dim=1))

class PSALoss(nn.Module):
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        # [수정됨] 이중 가중치 제거! WeightedRandomSampler만으로 밸런스 유지
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    def forward(self, logits, labels): return self.ce(logits, labels)

def batch_to_device(batch, device): return {k: v.to(device) for k, v in batch.items()}

def run_fold(fold, file_list, args, device):
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = build_dataloaders(file_list, fold, args.batch_size, args.num_workers)
    model, criterion = PSAGradingModel().to(device), PSALoss().to(device)
    
    optimizer = torch.optim.AdamW([
        {"params": list(model.enc_full.parameters()) + list(model.enc_corners.parameters()) + list(model.enc_edges.parameters()) + list(model.enc_surface.parameters()), "lr": 1e-4},
        {"params": list(model.transformer.parameters()) + [model.pos_embed] + list(model.head.parameters()), "lr": 3e-4}
    ], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler()
    
    best_auc, patience_cnt = 0.0, 0
    for epoch in range(args.epochs):
        model.train(); tr_loss, tr_correct, total = 0, 0, 0
        for batch in train_loader:
            batch = batch_to_device(batch, device); labels = batch["label"]
            optimizer.zero_grad()
            with autocast():
                logits = model(batch["full"], batch["corners"], batch["edges"], batch["surface"])
                loss = criterion(logits, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            tr_loss += loss.item() * labels.size(0); tr_correct += (logits.argmax(1) == labels).sum().item(); total += labels.size(0)
        scheduler.step()
        
        model.eval(); vl_loss, vl_correct, v_total, all_probs, all_labels = 0, 0, 0, [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch_to_device(batch, device); labels = batch["label"]
                with autocast():
                    logits = model(batch["full"], batch["corners"], batch["edges"], batch["surface"])
                    loss = criterion(logits, labels)
                vl_loss += loss.item() * labels.size(0); vl_correct += (logits.argmax(1) == labels).sum().item(); v_total += labels.size(0)
                all_probs.extend(torch.softmax(logits, dim=-1)[:, 2].cpu().numpy()); all_labels.extend((labels == 2).cpu().numpy().astype(int))
        
        try: val_auc = roc_auc_score(all_labels, all_probs)
        except: val_auc = 0.0
        
        print(f"[Fold {fold} | Ep {epoch+1:02d}/{args.epochs}] Tr_Loss: {tr_loss/total:.4f} Tr_Acc: {tr_correct/total:.4f} | Vl_Loss: {vl_loss/v_total:.4f} Vl_Acc: {vl_correct/v_total:.4f} | AUC(10점): {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc, patience_cnt = val_auc, 0
            torch.save(model.state_dict(), save_dir / f"best_fold{fold}.pth")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience: print(f"🛑 [Early stopping] Fold {fold}"); break
            
    # 모의고사 성적표 즉시 출력 기능 추가
    model.load_state_dict(torch.load(save_dir / f"best_fold{fold}.pth"))
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch_to_device(batch, device)
            val_preds.extend(model(batch["full"], batch["corners"], batch["edges"], batch["surface"]).argmax(1).cpu().tolist())
            val_true.extend(batch["label"].cpu().tolist())
    print(f"\n🎯 [Fold {fold} 모의고사 성적표 (Classes: 8, 9, 10)] 🎯\n" + classification_report(val_true, val_preds, target_names=["PSA 8", "PSA 9", "PSA 10"], digits=4))
    
    return {"fold": fold, "auc": best_auc}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 경로 기본값 맞춤
    parser.add_argument("--data_root", type=str, default="/data3/home/h22000561/psa_grading/data/train_yolo")
    parser.add_argument("--save_dir", type=str, default="/data/EunJi/h22000561_psa/v21_tuned")
    parser.add_argument("--all_folds", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16) # A30 메모리에 맞게 16으로 조정
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 시작 (Device: {device})")
    
    file_list = load_file_list(args.data_root)
    if args.all_folds:
        for f in range(5): run_fold(f, file_list, args, device)
