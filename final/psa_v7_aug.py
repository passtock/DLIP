import os, re, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import timm
import torchvision.transforms as T
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

CFG = {
    'data_root'   : '/home/h22000561/psa_grading/data/processed_cropped',
    'grades'      : [8, 9, 10],
    'img_size'    : 300,
    'corner_size' : 96,
    'edge_size'   : 32,
    'batch_size'  : 24,
    'epochs'      : 30,
    'num_folds'   : 5,
    'lr'          : 3e-4,
    'weight_decay': 1e-4,
    'dropout'     : 0.4,
    'num_workers' : 2,
    'tta_n'       : 8,
}

LABEL_MAP   = {8: 0, 9: 1, 10: 2}
CLASS_NAMES = ['PSA 8', 'PSA 9', 'PSA 10']

def build_dataframe(data_root):
    records = []
    data_root = Path(data_root)
    for grade in CFG['grades']:
        folder = data_root / f'PSA{grade}'
        if not folder.exists():
            print(f'[WARNING] {folder} not found'); continue
            
        cert_dict = defaultdict(dict)
        for img_path in folder.glob('*.jpg'):
            match = re.match(r'(cert\d+)_PSA\d+_(front|back)', img_path.stem)
            if match:
                cert_dict[match.group(1)][match.group(2)] = str(img_path)
                
        for cert_id, sides in cert_dict.items():
            if 'front' in sides and 'back' in sides:
                records.append({
                    'cert_id': cert_id,
                    'front'  : sides['front'],
                    'back'   : sides['back'],
                    'grade'  : grade,
                    'label'  : LABEL_MAP[grade]
                })
                
    df = pd.DataFrame(records)
    print(f'Total cards: {len(df)}')
    return df

df = build_dataframe(CFG['data_root'])

def crop_regions(img: Image.Image) -> dict:
    W, H  = img.size
    cs    = CFG['corner_size']
    ew    = CFG['img_size']
    eh    = CFG['edge_size']
    return {
        'full'   : img,
        'tl'     : img.crop((0,    0,    cs,   cs)),
        'tr'     : img.crop((W-cs, 0,    W,    cs)),
        'bl'     : img.crop((0,    H-cs, cs,   H)),
        'br'     : img.crop((W-cs, H-cs, W,    H)),
        'top'    : img.crop(((W-ew)//2, 0,    (W+ew)//2, eh)),
        'bottom' : img.crop(((W-ew)//2, H-eh, (W+ew)//2, H)),
        'left'   : img.crop((0,    (H-ew)//2, eh,   (H+ew)//2)),
        'right'  : img.crop((W-eh, (H-ew)//2, W,    (H+ew)//2)),
        'surface': img.crop((W//4, H//4, 3*W//4, 3*H//4)),
    }

def make_tf(size, mode='train'):
    norm = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    if mode == 'train':
        return T.Compose([
            T.Resize((size, size)),
            # Key modification: Randomly shift image by up to 3% (X and Y) and rotate by up to 2 degrees
            T.RandomAffine(degrees=2, translate=(0.03, 0.03)),
            T.RandomHorizontalFlip(0.3),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
            T.ToTensor(), norm
        ])
    return T.Compose([T.Resize((size, size)), T.ToTensor(), norm])

class PSADataset(Dataset):
    def __init__(self, df, mode='train'):
        self.df   = df.reset_index(drop=True)
        self.mode = mode
        s  = CFG['img_size']
        cs = CFG['corner_size']
        es = CFG['edge_size']
        self.tf_full    = make_tf(s,    mode)
        self.tf_corner  = make_tf(cs,   mode)
        self.tf_edge    = make_tf(es,   mode)
        self.tf_surface = make_tf(s//2, mode)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        front = Image.open(row['front']).convert('RGB')
        back  = Image.open(row['back']).convert('RGB')
        cf, cb = crop_regions(front), crop_regions(back)

        full = torch.cat([
            self.tf_full(cf['full']), self.tf_full(cb['full'])
        ], dim=0)

        corners = torch.stack([
            self.tf_corner(cf['tl']), self.tf_corner(cf['tr']),
            self.tf_corner(cf['bl']), self.tf_corner(cf['br']),
            self.tf_corner(cb['tl']), self.tf_corner(cb['tr']),
            self.tf_corner(cb['bl']), self.tf_corner(cb['br']),
        ]).reshape(24, CFG['corner_size'], CFG['corner_size'])

        edges = torch.stack([
            self.tf_edge(cf['top']),    self.tf_edge(cf['bottom']),
            self.tf_edge(cf['left']),   self.tf_edge(cf['right']),
            self.tf_edge(cb['top']),    self.tf_edge(cb['bottom']),
            self.tf_edge(cb['left']),   self.tf_edge(cb['right']),
        ]).reshape(24, CFG['edge_size'], CFG['edge_size'])

        surface = torch.cat([
            self.tf_surface(cf['surface']), self.tf_surface(cb['surface'])
        ], dim=0)

        return full, corners, edges, surface, int(row['label'])

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight    = weight

    def forward(self, logits, labels):
        n_cls = logits.size(1)
        with torch.no_grad():
            soft = torch.full_like(logits, self.smoothing / (n_cls - 1))
            soft.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)
        log_prob = F.log_softmax(logits, dim=1)
        loss = -(soft * log_prob).sum(dim=1)
        if self.weight is not None:
            loss = loss * self.weight[labels]
        return loss.mean()

class RegionEncoder(nn.Module):
    def __init__(self, in_channels, out_dim=128, backbone='efficientnet_b0'):
        super().__init__()
        base = timm.create_model(backbone, pretrained=True)
        old  = base.conv_stem
        new_conv = nn.Conv2d(
            in_channels, old.out_channels,
            old.kernel_size, old.stride, old.padding, bias=False
        )
        with torch.no_grad():
            for i in range(in_channels):
                new_conv.weight[:, i] = old.weight[:, i % 3] / (in_channels / 3)
        base.conv_stem = new_conv
        feat_dim = base.classifier.in_features
        base.classifier = nn.Sequential(
            nn.Linear(feat_dim, out_dim), nn.ReLU()
        )
        self.encoder = base

    def forward(self, x):
        return self.encoder(x)

class PSAMultiBranchModel(nn.Module):
    def __init__(self, num_classes=3, feat_dim=128, dropout=0.4):
        super().__init__()
        self.centering_branch = RegionEncoder(6,  feat_dim)
        self.corner_branch    = RegionEncoder(24, feat_dim)
        self.edge_branch      = RegionEncoder(24, feat_dim)
        self.surface_branch   = RegionEncoder(6,  feat_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 4, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, full, corners, edges, surface):
        feat_full    = self.centering_branch(full)
        feat_corners = self.corner_branch(corners)
        feat_edges   = self.edge_branch(edges)
        feat_surface = self.surface_branch(surface)
        
        combined = torch.cat([feat_full, feat_corners, feat_edges, feat_surface], dim=1)
        return self.classifier(combined)

def compute_class_weights(df):
    counts  = df['label'].value_counts().sort_index()
    total   = len(df)
    weights = []
    for i in range(3):
        weights.append(total / (3 * counts[i]))
    weights = torch.FloatTensor(weights).to(DEVICE)
    return weights

class_weights = compute_class_weights(df)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = correct = total = 0
    for full, corners, edges, surface, labels in loader:
        full, corners, edges, surface, labels = (
            full.to(DEVICE), corners.to(DEVICE),
            edges.to(DEVICE), surface.to(DEVICE), labels.to(DEVICE)
        )
        optimizer.zero_grad()
        logits = model(full, corners, edges, surface)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels, all_probs = [], [], []
    for full, corners, edges, surface, labels in loader:
        full, corners, edges, surface, labels = (
            full.to(DEVICE), corners.to(DEVICE),
            edges.to(DEVICE), surface.to(DEVICE), labels.to(DEVICE)
        )
        logits = model(full, corners, edges, surface)
        loss   = criterion(logits, labels)
        probs  = F.softmax(logits, dim=1)

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += len(labels)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    return total_loss/total, correct/total, all_preds, all_labels, all_probs

@torch.no_grad()
def tta_predict(model, full, corners, edges, surface, n=8):
    model.eval()
    probs_list = []

    def augment(x, aug_idx):
        if aug_idx == 0: return x
        if aug_idx == 1: return torch.flip(x, dims=[-1])
        if aug_idx == 2: return torch.flip(x, dims=[-2])
        if aug_idx == 3: return torch.rot90(x, 1, dims=[-2,-1])
        if aug_idx == 4: return torch.rot90(x, 2, dims=[-2,-1])
        if aug_idx == 5: return torch.rot90(x, 3, dims=[-2,-1])
        if aug_idx == 6: return torch.rot90(torch.flip(x, dims=[-1]), 1, dims=[-2,-1])
        if aug_idx == 7: return torch.rot90(torch.flip(x, dims=[-2]), 1, dims=[-2,-1])

    for i in range(n):
        logit = model(augment(full,i), augment(corners,i),
                      augment(edges,i), augment(surface,i))
        probs_list.append(F.softmax(logit, dim=1))
    return torch.stack(probs_list).mean(0)

skf = StratifiedKFold(n_splits=CFG['num_folds'], shuffle=True, random_state=SEED)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label']), 1):
    print(f'\n--- Fold {fold}/{CFG["num_folds"]} ---')

    train_ds = PSADataset(df.iloc[train_idx], mode='train')
    val_ds   = PSADataset(df.iloc[val_idx],   mode='val')
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                              shuffle=True, num_workers=CFG["num_workers"], drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG['batch_size'],
                              shuffle=False, num_workers=CFG['num_workers'])

    model     = PSAMultiBranchModel().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG['epochs'], eta_min=1e-6)
    criterion = LabelSmoothingCE(smoothing=0.1, weight=class_weights)

    best_acc = 0.0
    history  = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}

    for epoch in range(1, CFG['epochs']+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, _, _, _ = validate(model, val_loader, criterion)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        marker = ' <- best' if vl_acc > best_acc else ''
        print(f'  Ep {epoch:02d} | Train {tr_loss:.4f}/{tr_acc:.4f} | Val {vl_loss:.4f}/{vl_acc:.4f}{marker}')

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), f'best_fold{fold}.pth')

    fold_results.append({'fold': fold, 'best_acc': best_acc, 'history': history})

_, val_idx_last = list(skf.split(df, df['label']))[CFG['num_folds']-1]
val_ds_eval     = PSADataset(df.iloc[val_idx_last], mode='val')
val_loader_eval = DataLoader(val_ds_eval, batch_size=CFG['batch_size'],
                             shuffle=False, num_workers=CFG['num_workers'])

all_models = []
for fold in range(1, CFG['num_folds']+1):
    m = PSAMultiBranchModel().to(DEVICE)
    m.load_state_dict(torch.load(f'best_fold{fold}.pth', map_location=DEVICE))
    m.eval()
    all_models.append(m)

all_probs_ensemble, all_labels_eval = [], []
with torch.no_grad():
    for full, corners, edges, surface, labels in val_loader_eval:
        full, corners, edges, surface = (
            full.to(DEVICE), corners.to(DEVICE),
            edges.to(DEVICE), surface.to(DEVICE)
        )
        probs_list = [
            tta_predict(m, full, corners, edges, surface, n=CFG['tta_n'])
            for m in all_models
        ]
        avg_probs = torch.stack(probs_list).mean(0)
        all_probs_ensemble.extend(avg_probs.cpu().tolist())
        all_labels_eval.extend(labels.tolist())

all_preds_ensemble = []
for p in all_probs_ensemble:
    if p[2] > p[0] and p[2] > p[1]:
        all_preds_ensemble.append(2)
    else:
        all_preds_ensemble.append(0 if p[0] >= p[1] else 1)

print('\n=== Performance by Conservative Margin Penalty ===')
for margin in [1.1, 1.3, 1.5, 2.0]:
    preds_t = []
    for p in all_probs_ensemble:
        if p[2] > (p[1] * margin) and p[2] > (p[0] * margin):
            preds_t.append(2)
        else:
            preds_t.append(0 if p[0] >= p[1] else 1)
            
    acc = accuracy_score(all_labels_eval, preds_t)
    f1  = f1_score(all_labels_eval, preds_t, average='macro')
    print(f'  margin={margin:.1f} | Acc={acc:.4f} | Macro-F1={f1:.4f}')

