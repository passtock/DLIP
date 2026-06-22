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
# 1. 설정 및 하이퍼파라미터 (RAW 경로 지정)
# ==========================================
RES_DIR = '/data/EunJi/h22000561_psa' 
DATA_ROOT = '/data3/home/h22000561/psa_grading/data/raw' # 요청하신 Raw 데이터 경로

os.makedirs(RES_DIR, exist_ok=True)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'>>> Running Ultimate RAW Master on: {DEVICE} <<<')

CFG = {
    'img_size'    : 300,
    'corner_size' : 96,  # 고해상도에서 추출할 모서리 크기
    'edge_size'   : 32,  # 고해상도에서 추출할 테두리 두께
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

# RAW 이미지에서 겉면 플라스틱을 걷어내고 카드만 남기는 '비율' 크롭 좌표 (해상도 압축 없음)
TRAIN_CROP_RATIO = {'rx': 0.0974, 'ry': 0.2613, 'rw': 0.9026, 'rh': 0.9280}

# ==========================================
# 2. 데이터 파이프라인 (RAW 폴더 정규식 파싱)
# ==========================================
def build_dataframe(data_root):
    records = []
    data_root = Path(data_root)
    # PSA8, PSA9, PSA10 폴더 순회
    for grade in [8, 9, 10]:
        folder = data_root / f'PSA{grade}'
        if not folder.exists(): continue
            
        cert_dict = defaultdict(dict)
        for img_path in folder.glob('*.jpg'):
            fn = img_path.stem.lower()
            match_cert = re.search(r'(cert\d+)', fn) # RAW 데이터 파일명에서 cert 번호 추출
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
    print(f"Total RAW cards loaded: {len(df)}")
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
        
        # [핵심] 리사이즈(압축) 절.대.없.음! 고화질 비율 크롭 상태 그대로 반환
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
# 3. 모델 아키텍처 정의 (AttentionFusion 포함)
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
        features = torch.stack([self.centering(f), self.corner(c), self.edge(e), self.surface(s)], dim=1)
        fused = self.attention_fusion(features)
        return self.classifier(fused.reshape(fused.size(0), -1))

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
# 4. 5-Fold 학습 및 검증 루프
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
                    loss = criterion(out, l)
                    vl_loss += loss.item() * len(l); vl_acc += (out.argmax(1) == l).sum().item()
            
            vl_loss /= len(val_df); vl_acc /= len(val_df)
            
            fold_history['train_loss'].append(tr_loss); fold_history['train_acc'].append(tr_acc)
            fold_history['val_loss'].append(vl_loss); fold_history['val_acc'].append(vl_acc)
            
            if vl_acc > best_acc:
                best_acc = vl_acc
                torch.save(ema.ema.state_dict(), f'{RES_DIR}/best_model_fold{fold}.pth')
                
            print(f'  Ep {epoch:02d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}')
            
        epochs_range = range(1, CFG['epochs'] + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, fold_history['train_loss'], label='Train Loss')
        plt.plot(epochs_range, fold_history['val_loss'], label='Val Loss')
        plt.title(f'Fold {fold} - Loss History')
        plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, fold_history['train_acc'], label='Train Acc')
        plt.plot(epochs_range, fold_history['val_acc'], label='Val Acc')
        plt.title(f'Fold {fold} - Accuracy History')
        plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()
        plt.tight_layout()
        plt.savefig(f'{RES_DIR}/fold{fold}_training_curves.png', dpi=150)
        plt.close()
            
        del model, ema, optimizer; gc.collect(); torch.cuda.empty_cache()

    # ==========================================
    # 5. 최종 앙상블 리포트 및 시각화 저장
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
    target_probs = all_probs[:, 1]

    with open(f'{RES_DIR}/academic_report.txt', 'w', encoding='utf-8') as f_out:
        f_out.write("=== Final Binary Ensemble Report (Threshold 0.50) ===\n")
        f_out.write(classification_report(all_labels, preds_05, target_names=CLASS_NAMES, digits=4))
        f_out.write("\n\n=== [Business Optimized] Report (Threshold 0.70) ===\n")
        f_out.write(classification_report(all_labels, preds_07, target_names=CLASS_NAMES, digits=4))

    print("\n" + "="*50 + "\n=== [RAW 고화질 마스터 완료] 최종 앙상블 리포트 ===\n" + "="*50)
    print(classification_report(all_labels, preds_05, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(all_labels, preds_05)
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Threshold 0.50)')
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES); plt.yticks(tick_marks, CLASS_NAMES)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Actual label'); plt.xlabel('Predicted label')

    fpr, tpr, _ = roc_curve(all_labels, target_probs)
    plt.subplot(1, 3, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc(fpr, tpr):.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.legend(loc="lower right")

    precision, recall, _ = precision_recall_curve(all_labels, target_probs)
    plt.subplot(1, 3, 3)
    plt.plot(recall, precision, color='green', lw=2, label=f'PR Curve (AUC = {auc(recall, precision):.4f})')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(f'{RES_DIR}/final_ensemble_evaluation_plots.png', dpi=150)
    plt.close()

    print(f"\n🎯 [시각화 포함 완료] 결과 리포트와 분석 그래프가 저장되었습니다: {RES_DIR}/")
