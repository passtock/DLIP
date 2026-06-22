import os
import re
import random
import gc
from pathlib import Path
from collections import defaultdict
import copy
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import timm
import torchvision.transforms as T
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

# ==========================================
# 1. 설정 및 하이퍼파라미터 (서버 경로 반영 완료)
# ==========================================
RES_DIR = '/data/EunJi/h22000561_psa' 
DATA_ROOT = '/home/h22000561/psa_grading/data/processed_cropped'  # 알려주신 경로로 수정 완료

os.makedirs(RES_DIR, exist_ok=True)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'>>> Running Ultimate Master Version v13 on: {DEVICE} <<<')

CFG = {
    'img_size'    : 300,
    'corner_size' : 96,
    'edge_size'   : 32,
    'batch_size'  : 16,        
    'epochs'      : 25,        
    'num_folds'   : 5,         
    'lr'          : 2e-4,      
    'weight_decay': 1e-2,      
    'dropout'     : 0.3,       
    'num_workers' : 4,         
    'label_smoothing': 0.05    
}
CLASS_NAMES = ['Non-Gem (8,9)', 'Gem Mint (10)']

# ==========================================
# 2. 데이터 파이프라인
# ==========================================
def build_dataframe(data_root):
    records = []
    data_root = Path(data_root)
    for grade in [8, 9, 10]:
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
                    'label': 1 if grade == 10 else 0
                })
    df = pd.DataFrame(records)
    print(f"Total cards loaded: {len(df)}")
    return df

df = build_dataframe(DATA_ROOT)

def crop_regions(img):
    W, H = img.size
    cs = CFG['corner_size']
    ew = CFG['img_size']
    eh = CFG['edge_size']
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
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
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

# ==========================================
# 3. 모델 정의
# ==========================================
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

class AttentionFusion(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        return self.proj(attn @ v)

class PSAMultiBranchModel(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.centering = RegionEncoder(6, embed_dim)
        self.corner = RegionEncoder(24, embed_dim)
        self.edge = RegionEncoder(24, embed_dim)
        self.surface = RegionEncoder(6, embed_dim)
        
        self.attention_fusion = AttentionFusion(dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(CFG['dropout']),
            nn.Linear(embed_dim * 4, 128),
            nn.GELU(),
            nn.Dropout(CFG['dropout']/2),
            nn.Linear(128, 2)
        )
        
    def forward(self, f, c, e, s):
        feat_f = self.centering(f)
        feat_c = self.corner(c)
        feat_e = self.edge(e)
        feat_s = self.surface(s)
        
        features = torch.stack([feat_f, feat_c, feat_e, feat_s], dim=1)
        fused = self.attention_fusion(features)
        fused_flat = fused.reshape(fused.size(0), -1)
        return self.classifier(fused_flat)

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for param in self.ema.parameters(): param.requires_grad = False
    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.ema.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(self.decay * ema_v + (1. - self.decay) * model_v)

# ==========================================
# 4. 학습 및 검증 루프
# ==========================================
if __name__ == '__main__':
    skf = StratifiedKFold(n_splits=CFG['num_folds'], shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label']), 1):
        print(f'\n--- Training Fold {fold}/{CFG["num_folds"]} ---')
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)
        
        class_counts = train_df['label'].value_counts().sort_index().values
        class_weights_sampler = 1.0 / class_counts
        sample_weights = [class_weights_sampler[label] for label in train_df['label'].values]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            
        train_loader = DataLoader(PSADataset(train_df, 'train'), batch_size=CFG['batch_size'], sampler=sampler, num_workers=CFG['num_workers'], drop_last=True)
        val_loader   = DataLoader(PSADataset(val_df, 'val'), batch_size=CFG['batch_size'], shuffle=False, num_workers=CFG['num_workers'])
            
        model = PSAMultiBranchModel().to(DEVICE)
        ema = ModelEMA(model)
        
        optimizer = AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        scheduler = OneCycleLR(optimizer, max_lr=CFG['lr'], steps_per_epoch=len(train_loader), epochs=CFG['epochs'])
        criterion = nn.CrossEntropyLoss(label_smoothing=CFG['label_smoothing'])
        
        best_acc = 0.0
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
                scheduler.step()
                ema.update(model)
                
                tr_loss += loss.item() * len(l); tr_acc += (out.argmax(1) == l).sum().item()
            
            tr_loss /= len(train_df); tr_acc /= len(train_df)
            
            ema.ema.eval()
            vl_loss = vl_acc = 0
            with torch.no_grad():
                for f, c, e, s, l in val_loader:
                    f, c, e, s, l = f.to(DEVICE), c.to(DEVICE), e.to(DEVICE), s.to(DEVICE), l.to(DEVICE)
                    out = ema.ema(f, c, e, s)
                    loss = criterion(out, l)
                    vl_loss += loss.item() * len(l); vl_acc += (out.argmax(1) == l).sum().item()
            
            vl_loss /= len(val_df); vl_acc /= len(val_df)
            
            if vl_acc > best_acc:
                best_acc = vl_acc
                torch.save(ema.ema.state_dict(), f'{RES_DIR}/best_model_fold{fold}.pth')
                
            print(f'  Ep {epoch:02d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}')
            
        del model, ema, optimizer; gc.collect(); torch.cuda.empty_cache()

    # ==========================================
    # 5. 최종 앙상블 리포트 저장
    # ==========================================
    print("\n>>> Running Full Dataset Ensemble Evaluation... <<<")
    final_full_loader = DataLoader(PSADataset(df, 'val'), batch_size=CFG['batch_size'], shuffle=False, num_workers=CFG['num_workers'])
    models = []
    for fold in range(1, CFG['num_folds'] + 1):
        m = PSAMultiBranchModel().to(DEVICE)
        m.load_state_dict(torch.load(f'{RES_DIR}/best_model_fold{fold}.pth'))
        m.eval()
        models.append(m)

    def apply_tta(tensor, tta_idx):
        if tta_idx == 0: return tensor
        elif tta_idx == 1: return torch.flip(tensor, [-1])
        elif tta_idx == 2: return torch.flip(tensor, [-2])
        return tensor

    all_probs, all_labels = [], []
    with torch.no_grad():
        for f, c, e, s, l in final_full_loader:
            f, c, e, s = f.to(DEVICE), c.to(DEVICE), e.to(DEVICE), s.to(DEVICE)
            probs_list = []
            for m in models:
                for aug_idx in range(3):
                    ff, cc, ee, ss = apply_tta(f, aug_idx), apply_tta(c, aug_idx), apply_tta(e, aug_idx), apply_tta(s, aug_idx)
                    probs_list.append(F.softmax(m(ff, cc, ee, ss), dim=1))
            all_probs.extend(torch.stack(probs_list).mean(0).cpu().tolist())
            all_labels.extend(l.tolist())

    all_probs = np.array(all_probs)
    preds_05 = np.argmax(all_probs, axis=1)
    preds_07 = np.where(all_probs[:, 1] >= 0.70, 1, 0)

    with open(f'{RES_DIR}/academic_report.txt', 'w', encoding='utf-8') as f_out:
        f_out.write("=== Final Binary Ensemble Report (Threshold 0.50) ===\n")
        f_out.write(classification_report(all_labels, preds_05, target_names=CLASS_NAMES))
        f_out.write("\n\n=== [Business Optimized] Report (Threshold 0.70) ===\n")
        f_out.write(classification_report(all_labels, preds_07, target_names=CLASS_NAMES))

    print(f"\n>>> 완료! 결과 보고서 파일이 저장되었습니다: {RES_DIR}/academic_report.txt")
