import os, re, random, gc

# FIXED: Completely disable Hugging Face file locking mechanism to prevent multi-branch loop hangs
os.environ["HF_HUB_DISABLE_LOCKING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TORCH_HUB_OFFLINE"] = "1"

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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

# Setup output directory
RES_DIR = 'final_project_v10_results'
os.makedirs(RES_DIR, exist_ok=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'>>> Running Pure Offline Cross-Validation on: {DEVICE} <<<')

CFG = {
    'data_root'   : '/home/h22000561/psa_grading/data/processed_cropped',
    'grades'      : [8, 9, 10],
    'img_size'    : 300,
    'corner_size' : 96,
    'edge_size'   : 32,
    'batch_size'  : 16,        
    'epochs'      : 25,        
    'num_folds'   : 5,         
    'lr'          : 1e-4,      
    'weight_decay': 5e-3,      
    'dropout'     : 0.5,       
    'num_workers' : 0,         
}

LABEL_MAP   = {8: 0, 9: 1, 10: 2}
CLASS_NAMES = ['PSA 8', 'PSA 9', 'PSA 10']

def build_dataframe(data_root):
    records = []
    data_root = Path(data_root)
    for grade in CFG['grades']:
        folder = data_root / f'PSA{grade}'
        if not folder.exists(): continue
        cert_dict = defaultdict(dict)
        for img_path in folder.glob('*.jpg'):
            match = re.match(r'(cert\d+)_PSA\d+_(front|back)', img_path.stem)
            if match:
                cert_dict[match.group(1)][match.group(2)] = str(img_path)
        for cert_id, sides in cert_dict.items():
            if 'front' in sides and 'back' in sides:
                records.append({
                    'front': sides['front'], 'back': sides['back'],
                    'label': LABEL_MAP[grade]
                })
    return pd.DataFrame(records)

df = build_dataframe(CFG['data_root'])

def crop_regions(img):
    W, H = img.size; cs = CFG['corner_size']; ew = CFG['img_size']; eh = CFG['edge_size']
    return {
        'full': img,
        'tl': img.crop((0, 0, cs, cs)), 'tr': img.crop((W-cs, 0, W, cs)),
        'bl': img.crop((0, H-cs, cs, H)), 'br': img.crop((W-cs, H-cs, W, H)),
        'top': img.crop(((W-ew)//2, 0, (W+ew)//2, eh)), 'bottom': img.crop(((W-ew)//2, H-eh, (W+ew)//2, H)),
        'left': img.crop((0, (H-ew)//2, eh, (H+ew)//2)), 'right': img.crop((W-eh, (H-ew)//2, W, (H+ew)//2)),
        'surface': img.crop((W//4, H//4, 3*W//4, 3*H//4)),
    }

def make_tf(size, mode='train'):
    norm = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    if mode == 'train':
        return T.Compose([
            T.Resize((size, size)),
            T.RandomAffine(degrees=2), 
            T.RandomHorizontalFlip(0.3),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
            T.ToTensor(), norm
        ])
    return T.Compose([T.Resize((size, size)), T.ToTensor(), norm])

class PSADataset(Dataset):
    def __init__(self, target_df, mode='train'):
        self.df = target_df.reset_index(drop=True)
        self.tf_full = make_tf(CFG['img_size'], mode)
        self.tf_corner = make_tf(CFG['corner_size'], mode)
        self.tf_edge = make_tf(CFG['edge_size'], mode)
        self.tf_surface = make_tf(CFG['img_size']//2, mode)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        front = Image.open(row['front']).convert('RGB')
        back = Image.open(row['back']).convert('RGB')
        cf, cb = crop_regions(front), crop_regions(back)

        full = torch.cat([self.tf_full(cf['full']), self.tf_full(cb['full'])], dim=0)
        corners = torch.stack([
            self.tf_corner(cf['tl']), self.tf_corner(cf['tr']), self.tf_corner(cf['bl']), self.tf_corner(cf['br']),
            self.tf_corner(cb['tl']), self.tf_corner(cb['tr']), self.tf_corner(cb['bl']), self.tf_corner(cb['br'])
        ]).reshape(24, CFG['corner_size'], CFG['corner_size'])
        edges = torch.stack([
            self.tf_edge(cf['top']), self.tf_edge(cf['bottom']), self.tf_edge(cf['left']), self.tf_edge(cf['right']),
            self.tf_edge(cb['top']), self.tf_edge(cb['bottom']), self.tf_edge(cb['left']), self.tf_edge(cb['right'])
        ]).reshape(24, CFG['edge_size'], CFG['edge_size'])
        surface = torch.cat([self.tf_surface(cf['surface']), self.tf_surface(cb['surface'])], dim=0)

        return full, corners, edges, surface, int(row['label'])

class RegionEncoder(nn.Module):
    def __init__(self, in_channels, out_dim=128):
        super().__init__()
        base = timm.create_model('efficientnet_b2', pretrained=True)
        old = base.conv_stem
        new_conv = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            for i in range(in_channels): new_conv.weight[:, i] = old.weight[:, i % 3] / (in_channels / 3)
        base.conv_stem = new_conv
        base.classifier = nn.Sequential(nn.Linear(base.classifier.in_features, out_dim), nn.ReLU())
        self.encoder = base
    def forward(self, x): return self.encoder(x)

class PSAMultiBranchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.centering = RegionEncoder(6)
        self.corner = RegionEncoder(24)
        self.edge = RegionEncoder(24)
        self.surface = RegionEncoder(6)
        self.classifier = nn.Sequential(
            nn.Dropout(CFG['dropout']), nn.Linear(512, 256), nn.GELU(),
            nn.Dropout(CFG['dropout']/2), nn.Linear(256, 3)
        )
    def forward(self, f, c, e, s):
        return self.classifier(torch.cat([self.centering(f), self.corner(c), self.edge(e), self.surface(s)], dim=1))

skf = StratifiedKFold(n_splits=CFG['num_folds'], shuffle=True, random_state=SEED)
fold_histories = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label']), 1):
    print(f'\n--- Training Fold {fold}/{CFG["num_folds"]} ---')
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)
    
    class_counts = train_df['label'].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_df['label'].values]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(PSADataset(train_df, 'train'), batch_size=CFG['batch_size'], sampler=sampler, num_workers=CFG['num_workers'])
    val_loader   = DataLoader(PSADataset(val_df, 'val'), batch_size=CFG['batch_size'], shuffle=False, num_workers=CFG['num_workers'])
    
    model = PSAMultiBranchModel().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG['epochs'], eta_min=1e-6)
    criterion = nn.CrossEntropyLoss() 
    
    best_acc = 0.0
    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    
    for epoch in range(1, CFG['epochs'] + 1):
        model.train()
        tr_loss = tr_acc = 0
        for f, c, e, s, l in train_loader:
            f, c, e, s, l = f.to(DEVICE), c.to(DEVICE), e.to(DEVICE), s.to(DEVICE), l.to(DEVICE)
            optimizer.zero_grad()
            out = model(f, c, e, s)
            loss = criterion(out, l)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(l); tr_acc += (out.argmax(1) == l).sum().item()
        
        tr_loss /= len(train_df); tr_acc /= len(train_df)
        
        model.eval()
        vl_loss = vl_acc = 0
        with torch.no_grad():
            for f, c, e, s, l in val_loader:
                f, c, e, s, l = f.to(DEVICE), c.to(DEVICE), e.to(DEVICE), s.to(DEVICE), l.to(DEVICE)
                out = model(f, c, e, s)
                loss = criterion(out, l)
                vl_loss += loss.item() * len(l); vl_acc += (out.argmax(1) == l).sum().item()
        
        vl_loss /= len(val_df); vl_acc /= len(val_df)
        scheduler.step()
        
        history['train_loss'].append(tr_loss); history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc); history['val_acc'].append(vl_acc)
        
        marker = ' <- best' if vl_acc > best_acc else ''
        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), f'{RES_DIR}/best_model_fold{fold}.pth')
            
        print(f'  Ep {epoch:02d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}{marker}')
        
    fold_histories.append(history)
    
    del model, optimizer, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

# Metric 1: Generate Overfitting-Check Curves
plt.figure(figsize=(12, 5))
for i, h in enumerate(fold_histories):
    plt.subplot(1, 2, 1)
    plt.plot(h['val_loss'], label=f'Fold {i+1}')
    plt.subplot(1, 2, 2)
    plt.plot(h['val_acc'], label=f'Fold {i+1}')
plt.subplot(1, 2, 1); plt.title('Validation Loss Curves'); plt.xlabel('Epochs'); plt.legend()
plt.subplot(1, 2, 2); plt.title('Validation Accuracy Curves'); plt.xlabel('Epochs'); plt.legend()
plt.tight_layout()
plt.savefig(f'{RES_DIR}/1_ensemble_learning_curves.png', dpi=150)
plt.close()

# Ensemble Evaluation
_, final_val_idx = list(skf.split(df, df['label']))[CFG['num_folds']-1]
final_val_loader = DataLoader(PSADataset(df.iloc[final_val_idx], 'val'), batch_size=CFG['batch_size'], shuffle=False, num_workers=CFG['num_workers'])

models = []
for fold in range(1, CFG['num_folds'] + 1):
    m = PSAMultiBranchModel().to(DEVICE)
    m.load_state_dict(torch.load(f'{RES_DIR}/best_model_fold{fold}.pth'))
    m.eval()
    models.append(m)

all_probs, all_labels = [], []
with torch.no_grad():
    for f, c, e, s, l in final_val_loader:
        f, c, e, s = f.to(DEVICE), c.to(DEVICE), e.to(DEVICE), s.to(DEVICE)
        probs_list = []
        for m in models:
            for aug_idx in range(CFG['tta_n']):
                ff = f if aug_idx==0 else torch.flip(f, [-1])
                cc = c if aug_idx==0 else torch.flip(c, [-1])
                ee = e if aug_idx==0 else torch.flip(e, [-1])
                ss = s if aug_idx==0 else torch.flip(s, [-1])
                probs_list.append(F.softmax(m(ff, cc, ee, ss), dim=1))
        all_probs.extend(torch.stack(probs_list).mean(0).cpu().tolist())
        all_labels.extend(l.tolist())

all_preds = np.argmax(all_probs, axis=1)

with open(f'{RES_DIR}/academic_report.txt', 'w') as f:
    f.write("=== Final Academic Ensemble Report (Pure Argmax) ===\n")
    f.write(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# Metric 2: Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix (Academic)')
plt.tight_layout()
plt.savefig(f'{RES_DIR}/2_academic_confusion_matrix.png', dpi=150)
plt.close()

# Metric 3: ROC-AUC
auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
fpr, tpr, _ = roc_curve([1 if l==2 else 0 for l in all_labels], [p[2] for p in all_probs])
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, lw=2, label=f'Ensemble PSA 10 AUC = {auc:.4f}')
plt.plot([0,1], [0,1], '--', color='gray')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (Overall OvR AUC: {auc:.4f})')
plt.legend(); plt.tight_layout()
plt.savefig(f'{RES_DIR}/3_academic_roc_curve.png', dpi=150)
plt.close()

print(f"\n>>> Mission Accomplished! All pristine academic results saved in {RES_DIR}/ <<<")
