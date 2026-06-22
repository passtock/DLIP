import os
import re
import gc
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import timm
import torchvision.transforms as T
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

# ==========================================
# 1. 환경 설정 (A30 풀악셀 + 그래프 완벽 저장)
# ==========================================
RES_DIR = '/data/EunJi/h22000561_psa/v_ultimate_pro'
TRAIN_ROOT = '/data3/home/h22000561/psa_grading/data/raw'
TEST_ROOT = '/data3/home/h22000561/psa_grading/data/test'

os.makedirs(RES_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'>>> Running Ultimate PRO (A30 FULL THROTTLE 🚀) on: {DEVICE} <<<')

CFG = {
    'img_size'    : 512,  
    'batch_size'  : 16,         # [A30 전용] 24GB VRAM을 꽉 채우는 세팅
    'accum_steps' : 2,          # 사실상 Batch 32의 학습 효과
    'epochs'      : 25,   
    'num_folds'   : 5,         
    'lr'          : 2e-4,      
    'num_workers' : 8,          # [A30 전용] CPU 병목 제거
}
CLASS_NAMES = ['Non-Gem (8,9)', 'Gem Mint (10)']

# ==========================================
# 2. 데이터 파이프라인
# ==========================================
def load_data(data_root):
    records = []
    data_root = Path(data_root)
    for grade in [8, 9, 10]:
        folders = [data_root / f'PSA{grade}', data_root / f'psa_{grade}', data_root / f'psa{grade}']
        folder = next((f for f in folders if f.exists()), None)
        if not folder: continue
        cert_dict = defaultdict(dict)
        for img_path in folder.glob('*.jpg'):
            fn = img_path.stem.lower()
            match = re.search(r'(cert\d+)', fn)
            if match:
                cid = match.group(1)
                if 'front' in fn: cert_dict[cid]['front'] = str(img_path)
                elif 'back' in fn: cert_dict[cid]['back'] = str(img_path)
        for cert_id, sides in cert_dict.items():
            if 'front' in sides and 'back' in sides:
                records.append({'front': sides['front'], 'back': sides['back'], 'label': 1 if grade == 10 else 0})
    return pd.DataFrame(records)

train_df = load_data(TRAIN_ROOT)
test_df = load_data(TEST_ROOT)

def make_tf(mode='train'):
    norm = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    if mode == 'train':
        return T.Compose([
            T.Resize((CFG['img_size'], CFG['img_size'])),
            T.RandomHorizontalFlip(p=0.5), 
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.ToTensor(), norm,
            T.RandomErasing(p=0.2, scale=(0.02, 0.05))
        ])
    return T.Compose([T.Resize((CFG['img_size'], CFG['img_size'])), T.ToTensor(), norm])

class PSASiameseDataset(Dataset):
    def __init__(self, df, mode='train'):
        self.df = df.reset_index(drop=True)
        self.tf = make_tf(mode)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        f_img = Image.open(row['front']).convert('RGB')
        b_img = Image.open(row['back']).convert('RGB')
        return self.tf(f_img), self.tf(b_img), int(row['label'])

# ==========================================
# 3. 궁극의 아키텍처 & Focal Loss
# ==========================================
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        mask = self.sigmoid(self.conv(x))
        return x * mask 

class PSASiameseModel_PRO(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('convnext_small', pretrained=True, num_classes=0, global_pool='')
        dim = self.backbone.num_features
        
        self.spatial_attn_front = SpatialAttention(dim)
        self.spatial_attn_back = SpatialAttention(dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, front, back):
        feat_f = self.backbone(front)
        feat_b = self.backbone(back)
        
        attn_f = self.spatial_attn_front(feat_f)
        attn_b = self.spatial_attn_back(feat_b)
        
        pool_f = self.pool(attn_f).flatten(1)
        pool_b = self.pool(attn_b).flatten(1)
        
        return self.classifier(torch.cat([pool_f, pool_b], dim=1))

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha 
        self.gamma = gamma 
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * ((1 - pt) ** self.gamma) * ce_loss).mean()

# ==========================================
# 4. 원스톱 학습 및 테스트 루프
# ==========================================
if __name__ == '__main__':
    skf = StratifiedKFold(n_splits=CFG['num_folds'], shuffle=True, random_state=42)
    test_loader = DataLoader(PSASiameseDataset(test_df, 'val'), batch_size=1, shuffle=False, num_workers=4)
    scaler = torch.cuda.amp.GradScaler() 
    
    print("\n" + "="*50)
    print("🔥 [Phase 1] Ultimate PRO 5-Fold 학습 & 그래프 저장 시작 🔥")
    print("="*50)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label']), 1):
        print(f'\n--- Training Fold {fold}/{CFG["num_folds"]} ---')
        tr_df, vl_df = train_df.iloc[train_idx], train_df.iloc[val_idx]
        
        class_counts = tr_df['label'].value_counts().sort_index().values
        sampler = WeightedRandomSampler(weights=[(1.0 / class_counts)[lbl] for lbl in tr_df['label'].values], num_samples=len(tr_df), replacement=True)
            
        train_loader = DataLoader(PSASiameseDataset(tr_df, 'train'), batch_size=CFG['batch_size'], sampler=sampler, num_workers=CFG['num_workers'], drop_last=True)
        val_loader   = DataLoader(PSASiameseDataset(vl_df, 'val'), batch_size=CFG['batch_size'], shuffle=False, num_workers=CFG['num_workers'])
            
        model = PSASiameseModel_PRO().to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=CFG['lr'], weight_decay=2e-3)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        
        best_acc = 0.0
        
        # [핵심] 학습 기록을 저장할 딕셔너리 추가
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(1, CFG['epochs'] + 1):
            model.train()
            tr_loss = tr_acc = 0
            optimizer.zero_grad()
            
            for i, (f, b, l) in enumerate(train_loader):
                f, b, l = f.to(DEVICE), b.to(DEVICE), l.to(DEVICE)
                
                with torch.cuda.amp.autocast():
                    out = model(f, b)
                    loss = criterion(out, l) / CFG['accum_steps'] 
                
                scaler.scale(loss).backward()
                
                if (i + 1) % CFG['accum_steps'] == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                tr_loss += (loss.item() * CFG['accum_steps']) * len(l)
                tr_acc += (out.argmax(1) == l).sum().item()
            
            scheduler.step()
            tr_loss /= len(tr_df); tr_acc /= len(tr_df)
            
            model.eval()
            vl_loss = vl_acc = 0
            with torch.no_grad():
                for f, b, l in val_loader:
                    f, b, l = f.to(DEVICE), b.to(DEVICE), l.to(DEVICE)
                    with torch.cuda.amp.autocast():
                        out = model(f, b)
                        loss = criterion(out, l)
                    vl_loss += loss.item() * len(l); vl_acc += (out.argmax(1) == l).sum().item()
            vl_loss /= len(vl_df); vl_acc /= len(vl_df)
            
            # [핵심] 매 에포크마다 기록 저장
            history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
            history['val_loss'].append(vl_loss); history['val_acc'].append(vl_acc)
            
            if vl_acc > best_acc:
                best_acc = vl_acc
                torch.save(model.state_dict(), f'{RES_DIR}/best_model_fold{fold}.pth')
            print(f'  Ep {epoch:02d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}')
        
        # [핵심] 1개 폴드가 끝날 때마다 논문용 학습 곡선 그래프 저장!
        epochs_range = range(1, CFG['epochs'] + 1)
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
        plt.plot(epochs_range, history['val_loss'], label='Val Loss', marker='o')
        plt.title(f'Fold {fold} - Loss Curve (Focal Loss)')
        plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history['train_acc'], label='Train Accuracy', marker='o')
        plt.plot(epochs_range, history['val_acc'], label='Val Accuracy', marker='o')
        plt.title(f'Fold {fold} - Accuracy Curve')
        plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(f'{RES_DIR}/Fold_{fold}_Training_Curves.png', dpi=200)
        plt.close()
        print(f"  📊 Fold {fold} 학습 곡선 그래프가 저장되었습니다.")
            
        del model, optimizer; gc.collect(); torch.cuda.empty_cache()

    print("\n" + "="*50)
    print("🏆 [Phase 2] 실전 테스트셋(Test Set) 자동 앙상블 평가 🏆")
    print("="*50)

    models = []
    for fold in range(1, CFG['num_folds'] + 1):
        m = PSASiameseModel_PRO().to(DEVICE)
        m.load_state_dict(torch.load(f'{RES_DIR}/best_model_fold{fold}.pth'))
        m.eval()
        models.append(m)

    def apply_tta(t, i): return t if i == 0 else torch.flip(t, [-1]) if i == 1 else torch.flip(t, [-2])

    all_probs, all_labels = [], []
    with torch.no_grad():
        for idx, (f, b, l) in enumerate(test_loader, 1):
            f, b = f.to(DEVICE), b.to(DEVICE)
            probs_list = []
            for m in models:
                for a in range(3):
                    with torch.cuda.amp.autocast():
                        probs_list.append(F.softmax(m(apply_tta(f, a), apply_tta(b, a)), dim=1))
            
            all_probs.append(torch.stack(probs_list).mean(0).cpu().numpy()[0])
            all_labels.append(l.item())
            if idx % 20 == 0: print(f"  [실전 평가 중...] {idx}/{len(test_df)} 완료")

    all_probs = np.array(all_probs)
    preds = np.argmax(all_probs, axis=1)
    target_probs = all_probs[:, 1]
    
    report_txt = classification_report(all_labels, preds, target_names=CLASS_NAMES, digits=4)
    with open(f'{RES_DIR}/FINAL_ULTIMATE_PRO_REPORT.txt', 'w', encoding='utf-8') as f_out: f_out.write(report_txt)
    
    print("\n🎯 [ 궁극의 결론 ] 실전 테스트 성적표 🎯")
    print(report_txt)
    
    cm = confusion_matrix(all_labels, preds)
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1); plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues); plt.colorbar()
    for i, j in np.ndindex(cm.shape): plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > cm.max()/2. else "black")
    fpr, tpr, _ = roc_curve(all_labels, target_probs)
    plt.subplot(1, 3, 2); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Test AUC = {auc(fpr, tpr):.4f}'); plt.plot([0, 1], linestyle='--'); plt.legend()
    precision, recall, _ = precision_recall_curve(all_labels, target_probs)
    plt.subplot(1, 3, 3); plt.plot(recall, precision, color='green', lw=2, label=f'PR AUC = {auc(recall, precision):.4f}'); plt.legend()
    plt.tight_layout(); plt.savefig(f'{RES_DIR}/FINAL_EVALUATION_PLOTS.png', dpi=150); plt.close()
    
    print(f"✨ 모든 그래프와 성적표가 완벽하게 저장되었습니다: {RES_DIR}/")
