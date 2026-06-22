import os, re, random, argparse
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Parse Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--margin', type=float, default=1.3)
args = parser.parse_args()

# Create Results Directory
RES_DIR = f'results_lr{args.lr}_bs{args.batch_size}'
os.makedirs(RES_DIR, exist_ok=True)
print(f'>>> Starting Experiment: LR={args.lr}, Batch={args.batch_size} <<<')
print(f'Results will be saved in: {RES_DIR}/')

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CFG = {
    'data_root'   : '/home/h22000561/psa_grading/data/processed_cropped',
    'grades'      : [8, 9, 10],
    'img_size'    : 300,
    'corner_size' : 96,
    'edge_size'   : 32,
    'weight_decay': 1e-4,
    'dropout'     : 0.4,
    'num_workers' : 2,
    'tta_n'       : 4, 
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

# Split 80/20 (Single Fold for tuning)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=SEED)

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
    def __init__(self, df, mode='train'):
        self.df = df.reset_index(drop=True)
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

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing, self.weight = smoothing, weight
    def forward(self, logits, labels):
        n_cls = logits.size(1)
        with torch.no_grad():
            soft = torch.full_like(logits, self.smoothing / (n_cls - 1))
            soft.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(soft * F.log_softmax(logits, dim=1)).sum(dim=1)
        if self.weight is not None: loss = loss * self.weight[labels]
        return loss.mean()

class RegionEncoder(nn.Module):
    def __init__(self, in_channels, out_dim=128):
        super().__init__()
        base = timm.create_model('efficientnet_b0', pretrained=True)
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

train_loader = DataLoader(PSADataset(train_df, 'train'), batch_size=args.batch_size, shuffle=True, num_workers=CFG['num_workers'])
val_loader = DataLoader(PSADataset(val_df, 'val'), batch_size=args.batch_size, shuffle=False, num_workers=CFG['num_workers'])

model = PSAMultiBranchModel().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=CFG['weight_decay'])
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

weights = torch.FloatTensor([len(df)/(3*c) for c in df['label'].value_counts().sort_index()]).to(DEVICE)
criterion = LabelSmoothingCE(smoothing=0.1, weight=weights)

history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
best_acc = 0.0

for epoch in range(1, args.epochs + 1):
    model.train()
    tr_loss = tr_acc = 0
    for f, c, e, s, l in train_loader:
        f, c, e, s, l = f.to(DEVICE), c.to(DEVICE), e.to(DEVICE), s.to(DEVICE), l.to(DEVICE)
        optimizer.zero_grad()
        out = model(f, c, e, s)
        loss = criterion(out, l)
        loss.backward()
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
    
    if vl_acc > best_acc:
        best_acc = vl_acc
        torch.save(model.state_dict(), f'{RES_DIR}/best_model.pth')
    
    print(f'Ep {epoch:02d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}')

# METRIC 1: Loss Curve (Overfitting Check)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend(); plt.title('Loss Curve')
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.legend(); plt.title('Accuracy Curve')
plt.tight_layout()
plt.savefig(f'{RES_DIR}/1_loss_curve.png', dpi=150)
plt.close()

# Evaluation with TTA
model.load_state_dict(torch.load(f'{RES_DIR}/best_model.pth'))
model.eval()
all_probs, all_labels = [], []
with torch.no_grad():
    for f, c, e, s, l in val_loader:
        f, c, e, s = f.to(DEVICE), c.to(DEVICE), e.to(DEVICE), s.to(DEVICE)
        probs_list = []
        for aug_idx in range(CFG['tta_n']):
            ff = f if aug_idx==0 else torch.flip(f, [-1])
            cc = c if aug_idx==0 else torch.flip(c, [-1])
            ee = e if aug_idx==0 else torch.flip(e, [-1])
            ss = s if aug_idx==0 else torch.flip(s, [-1])
            probs_list.append(F.softmax(model(ff, cc, ee, ss), dim=1))
        all_probs.extend(torch.stack(probs_list).mean(0).cpu().tolist())
        all_labels.extend(l.tolist())

# Apply Margin Penalty
all_preds = []
for p in all_probs:
    if p[2] > (p[1] * args.margin) and p[2] > (p[0] * args.margin): all_preds.append(2)
    else: all_preds.append(0 if p[0] >= p[1] else 1)

# Report Generation
with open(f'{RES_DIR}/report.txt', 'w') as f:
    f.write(f"Hyperparameters: LR={args.lr}, Batch={args.batch_size}, Margin={args.margin}\n")
    f.write("="*50 + "\n")
    f.write(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# METRIC 2: Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(f'{RES_DIR}/2_confusion_matrix.png', dpi=150)
plt.close()

# METRIC 3: ROC-AUC
auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
fpr, tpr, _ = roc_curve([1 if l==2 else 0 for l in all_labels], [p[2] for p in all_probs])
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, lw=2, label=f'PSA 10 AUC = {auc:.4f}')
plt.plot([0,1], [0,1], '--', color='gray')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve for PSA 10 (Overall AUC: {auc:.4f})')
plt.legend()
plt.tight_layout()
plt.savefig(f'{RES_DIR}/3_roc_curve.png', dpi=150)
plt.close()

print(f">>> Finished! Results saved in {RES_DIR}/")
