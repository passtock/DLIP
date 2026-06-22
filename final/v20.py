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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

# ==========================================
# 1. 설정 및 하이퍼파라미터 (v20 밤샘용 고부하 세팅)
# ==========================================
RES_DIR = '/data/EunJi/h22000561_psa/v20' 
DATA_ROOT = '/data3/home/h22000561/psa_grading/data/raw'

os.makedirs(RES_DIR, exist_ok=True)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'>>> Running SOTA Masterpiece Version 20 on: {DEVICE} <<<')

CFG = {
    'img_size'    : 300,
    'corner_size' : 96,  
    'edge_size'   : 32,  
    'batch_size'  : 16,        
    'epochs'      : 35,  # 깊은 최적화를 위해 에포크 상향 조정 (밤샘 전용)
    'num_folds'   : 5,         
    'lr'          : 2e-4,      
    'weight_decay': 2e-2,  # 과적합 방지를 위해 가중치 감쇠 강화
    'dropout'     : 0.4,   # 규제 강화    
    'num_workers' : 4,         
    'label_smoothing': 0.05    
}
CLASS_NAMES = ['Non-Gem (8,9)', 'Gem Mint (10)']
TRAIN_CROP_RATIO = {'rx': 0.0974, 'ry': 0.2613, 'rw': 0.9026, 'rh': 0.9280}

# ==========================================
# 2. 데이터 파이프라인 (Random Erasing 오규제 도입)
# ==========================================
def build_dataframe(data_root):
    records = []
    data_root = Path(data_root)
    for grade in [8, 9, 10]:
        folder = data_root / f'PSA{grade}'
        if not folder.exists(): continue
        cert_dict = defaultdict(dict)
        for img_path in folder.glob('*.jpg'):
            fn = img_path.stem.lower()
            match_cert = re.search(r'(cert\d+)', fn)
            if match_cert:
                cid = match_cert.group(1)
                if 'front' in fn: cert_dict[cid]['front'] = str(img_path)
                elif 'back' in fn: cert_dict[cid]['back'] = str(img_path)
        for cert_id, sides in cert_dict.items():
            if 'front' in sides and 'back' in sides:
                records.append({
                    'front': sides['front'], 'back': sides['back'],
                    'label': 1 if grade == 10 else 0
                })
    df = pd.DataFrame(records)
    print(f"Total RAW cards loaded for v20: {len(df)}")
    return df

df = build_dataframe(DATA_ROOT)

def crop_regions(img):
    W, H = img.size
    cs, ew, eh = CFG['corner_size'], CFG['img_size'], CFG['edge_size']
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
            T.ToTensor(), norm,
            # [SOTA 기법] 픽셀 고착 과적합을 깨부수는 인공 결함 주입 스케일링
            T.RandomErasing(p=0.25, scale=(0.02, 0.08), value='random')
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
        def load_raw_and_crop(path):
            with Image.open(path).convert('RGB') as img:
                w, h = img.size
                return img.crop((int(TRAIN_CROP_RATIO['rx']*w), int(TRAIN_CROP_RATIO['ry']*h), int(TRAIN_CROP_RATIO['rw']*w), int(TRAIN_CROP_RATIO['rh']*h)))

        front = load_raw_and_crop(row['front'])
        back = load_raw_and_crop(row['back'])
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
# 3. 모델 아키텍처 (EfficientNet-B3 격상 + 2개층 Transformer)
# ==========================================
class RegionEncoder(nn.Module):
    def __init__(self, in_channels, out_dim=128):
        super().__init__()
        # 체급을 b2에서 b3로 묵직하게 빌드업
        base = timm.create_model('efficientnet_b3', pretrained=True)
        old = base.conv_stem
        new_conv = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            for i in range(in_channels): new_conv.weight[:, i] = old.weight[:, i % 3] / (in_channels / 3)
        base.conv_stem = new_conv
        base.classifier = nn.Sequential(nn.Linear(base.classifier.in_features, out_dim), nn.ReLU())
        self.encoder = base
    def forward(self, x): return self.encoder(x)

class RegionTransformerFusion(nn.Module):
    def __init__(self, dim=128, num_heads=4):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, 4, dim))
        # 트랜스포머 인코더 층을 2개 층으로 깊게 쌓아 특징 간 심층 융합 유도
        self.transformer = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=512, dropout=0.1, activation='gelu', batch_first=True),
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=512, dropout=0.1, activation='gelu', batch_first=True)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        x = x + self.pos_embed
        for layer in self.transformer:
            x = layer(x)
        return x.reshape(x.size(0), -1)

class PSAMultiBranchModel(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.centering = RegionEncoder(6, embed_dim)
        self.corner = RegionEncoder(24, embed_dim)
        self.edge = RegionEncoder(24, embed_dim)
        self.surface = RegionEncoder(6, embed_dim)
        
        self.transformer_fusion = RegionTransformerFusion(dim=embed_dim, num_heads=4)
        self.classifier = nn.Sequential(
            nn.Dropout(CFG['dropout']),
            nn.Linear(embed_dim * 4, 128),
            nn.GELU(),
            nn.Dropout(CFG['dropout']/2),
            nn.Linear(128, 2)
        )
        
    def forward(self, f, c, e, s):
        features = torch.stack([self.centering(f), self.corner(c), self.edge(e), self.surface(s)], dim=1)
        fused_flat = self.transformer_fusion(features)
        return self.classifier(fused_flat)

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for param in self.ema.parameters(): param.requires_grad = False
        self.decay = decay
    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.ema.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(self.decay * ema_v + (1. - self.decay) * model_v)

# ==========================================
# 4. 5-Fold 차등 하이퍼파라미터 학습 루프
# ==========================================
if __name__ == '__main__':
    skf = StratifiedKFold(n_splits=CFG['num_folds'], shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label']), 1):
        print(f'\n--- Training Fold {fold}/{CFG["num_folds"]} (v20 SOTA) ---')
        train_df, val_df = df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)
        
        class_counts = train_df['label'].value_counts().sort_index().values
        sampler = WeightedRandomSampler(weights=[(1.0 / class_counts)[lbl] for lbl in train_df['label'].values], num_samples=len(train_df), replacement=True)
            
        train_loader = DataLoader(PSADataset(train_df, 'train'), batch_size=CFG['batch_size'], sampler=sampler, num_workers=CFG['num_workers'], drop_last=True)
        val_loader   = DataLoader(PSADataset(val_df, 'val'), batch_size=CFG['batch_size'], shuffle=False, num_workers=CFG['num_workers'])
            
        model = PSAMultiBranchModel().to(DEVICE)
        ema = ModelEMA(model)
        
        # [SOTA 기법] 이미지 추출 특징 가중치 격리 차등 파라미터 셋업 (Backbone LR = Fusion LR / 10)
        backbone_params = []
        fusion_params = []
        for name, param in model.named_parameters():
            if any(k in name for k in ['centering', 'corner', 'edge', 'surface']):
                backbone_params.append(param)
            else:
                fusion_params.append(param)
                
        optimizer = AdamW([
            {'params': backbone_params, 'lr': CFG['lr'] * 0.1}, # Pretrained 파트는 아주 미세하게 정제 (2e-5)
            {'params': fusion_params, 'lr': CFG['lr']}          # 고차원 결합 파트는 메인 속도로 전개 (2e-4)
        ], weight_decay=CFG['weight_decay'])
        
        scheduler = OneCycleLR(optimizer, max_lr=[CFG['lr']*0.1, CFG['lr']], steps_per_epoch=len(train_loader), epochs=CFG['epochs'])
        criterion = nn.CrossEntropyLoss(label_smoothing=CFG['label_smoothing'])
        
        fold_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
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
                    vl_loss += criterion(out, l).item() * len(l); vl_acc += (out.argmax(1) == l).sum().item()
            
            vl_loss /= len(val_df); vl_acc /= len(val_df)
            fold_history['train_loss'].append(tr_loss); fold_history['train_acc'].append(tr_acc)
            fold_history['val_loss'].append(vl_loss); fold_history['val_acc'].append(vl_acc)
            
            if vl_acc > best_acc:
                best_acc = vl_acc
                torch.save(ema.ema.state_dict(), f'{RES_DIR}/best_model_fold{fold}.pth')
            print(f'  Ep {epoch:02d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}')
            
        epochs_range = range(1, CFG['epochs'] + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1); plt.plot(epochs_range, fold_history['train_loss'], label='Train Loss'); plt.plot(epochs_range, fold_history['val_loss'], label='Val Loss'); plt.legend()
        plt.subplot(1, 2, 2); plt.plot(epochs_range, fold_history['train_acc'], label='Train Acc'); plt.plot(epochs_range, fold_history['val_acc'], label='Val Acc'); plt.legend()
        plt.tight_layout(); plt.savefig(f'{RES_DIR}/fold{fold}_training_curves.png', dpi=150); plt.close()
        del model, ema, optimizer; gc.collect(); torch.cuda.empty_cache()

    # ==========================================
    # 5. 최종 앙상블 리포트 저장
    # ==========================================
    print("\n>>> Running Full Dataset Ensemble Evaluation (v20)... <<<")
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
        return torch.flip(tensor, [-2])

    all_probs, all_labels = [], []
    with torch.no_grad():
        for f, c, e, s, l in final_full_loader:
            f, c, e, s = f.to(DEVICE), c.to(DEVICE), e.to(DEVICE), s.to(DEVICE)
            probs_list = []
            for m in models:
                for aug_idx in range(3):
                    probs_list.append(F.softmax(m(apply_tta(f, aug_idx), apply_tta(c, aug_idx), apply_tta(e, aug_idx), apply_tta(s, aug_idx)), dim=1))
            all_probs.extend(torch.stack(probs_list).mean(0).cpu().tolist())
            all_labels.extend(l.tolist())

    all_probs = np.array(all_probs)
    preds_05 = np.argmax(all_probs, axis=1)
    target_probs = all_probs[:, 1]

    with open(f'{RES_DIR}/academic_report.txt', 'w', encoding='utf-8') as f_out:
        f_out.write("=== Final Binary Ensemble Report v20 (SOTA Masterpiece) ===\n")
        f_out.write(classification_report(all_labels, preds_05, target_names=CLASS_NAMES, digits=4))

    print("\n=== [v20 SOTA 마스터피스 완료] 최종 결과 ===")
    print(classification_report(all_labels, preds_05, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(all_labels, preds_05)
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1); plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues); plt.colorbar()
    for i, j in np.ndindex(cm.shape): plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > cm.max()/2. else "black")
    fpr, tpr, _ = roc_curve(all_labels, target_probs)
    plt.subplot(1, 3, 2); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc(fpr, tpr):.4f}'); plt.plot([0, 1], linestyle='--'); plt.legend()
    precision, recall, _ = precision_recall_curve(all_labels, target_probs)
    plt.subplot(1, 3, 3); plt.plot(recall, precision, color='green', lw=2, label=f'PR AUC = {auc(recall, precision):.4f}'); plt.legend()
    plt.tight_layout(); plt.savefig(f'{RES_DIR}/final_ensemble_evaluation_plots.png', dpi=150); plt.close()
    print(f"\n🎯 [밤샘 성과 완료] 결과 리포트가 저장되었습니다: {RES_DIR}/")
