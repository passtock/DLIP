import os, sys, re, gc
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
# 1. 환경 설정 (v3: 1만장 대규모 학습 + 과적합 방어)
# ==========================================
VERSION = "v_ultimate_pro_3"
RES_DIR = f'/data/EunJi/h22000561_psa/{VERSION}'
TRAIN_ROOT = '/data3/home/h22000561/psa_grading/data/train_yolo'
TEST_ROOT = '/data3/home/h22000561/psa_grading/data/test_yolo'

os.makedirs(RES_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CFG = {
    'img_size'    : 512,  # 512px 고해상도 통짜 이미지
    'batch_size'  : 16,   # A30 GPU VRAM에 맞춤
    'accum_steps' : 2,    # 사실상 배치 32의 효과
    'epochs'      : 30,   # 조기 종료가 있으므로 넉넉하게
    'num_folds'   : 5,         
    'lr'          : 2e-4,      
    'num_workers' : 8,          
}
CLASS_NAMES = ['Non-Gem (8,9)', 'Gem Mint (10)']

print("=======================================================")
print(f"🚀 Running {VERSION} (10K Dataset + EarlyStopping + SOTA) on {DEVICE}")
print("=======================================================")

# --- 조기 종료 (Early Stopping) 봇 ---
class EarlyStopping:
    def __init__(self, patience=5, path='best_model.pth'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'      [EarlyStopping] Caution: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

# ==========================================
# 2. 데이터 파이프라인
# ==========================================
def build_robust_dataframe(data_root):
    records = []
    data_root = Path(data_root)
    for grade in [8, 9, 10]:
        folder_candidates = [data_root / f'PSA{grade}', data_root / f'psa_{grade}', data_root / f'psa{grade}']
        folder = next((f for f in folder_candidates if f.exists()), None)
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

train_df = build_robust_dataframe(TRAIN_ROOT)
test_df = build_robust_dataframe(TEST_ROOT)

print(f"✅ 데이터 로드 완료 - 학습용: {len(train_df)}장 / 테스트용: {len(test_df)}장")

# --- 스파르타식 가혹한 데이터 증강 ---
def make_tf(mode='train'):
    norm = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    if mode == 'train':
        return T.Compose([
            T.Resize((CFG['img_size'], CFG['img_size'])),
            T.RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.RandomHorizontalFlip(p=0.5), 
            T.ToTensor(), norm,
            T.RandomErasing(p=0.3, scale=(0.02, 0.1), value='random') # 카드 일부를 가려서 억지로 응용력 키움
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
# 3. 모델 아키텍처 (Spatial Attention + ConvNeXt)
# ==========================================
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    def forward(self, x):
        return x * torch.sigmoid(self.conv(x))

class PSASiameseModel_PRO(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('convnext_small', pretrained=True, num_classes=0, global_pool='')
        dim = self.backbone.num_features
        self.spatial_attn_front = SpatialAttention(dim)
        self.spatial_attn_back = SpatialAttention(dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), # 강력한 드롭아웃
            nn.Linear(dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    def forward(self, front, back):
        feat_f, feat_b = self.backbone(front), self.backbone(back)
        pool_f = self.pool(self.spatial_attn_front(feat_f)).flatten(1)
        pool_b = self.pool(self.spatial_attn_back(feat_b)).flatten(1)
        return self.classifier(torch.cat([pool_f, pool_b], dim=1))

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * ((1 - pt) ** self.gamma) * ce_loss).mean()

# ==========================================
# 4. 학습 루프 (Training Loop)
# ==========================================
if __name__ == '__main__':
    skf = StratifiedKFold(n_splits=CFG['num_folds'], shuffle=True, random_state=42)
    scaler = torch.cuda.amp.GradScaler() 
    
    print("\n🔥 [Phase 1] v3 학습 시작 (과적합 원천 차단 모드) 🔥")
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label']), 1):
        print(f'\n--- Training Fold {fold}/{CFG["num_folds"]} ---')
        tr_df, vl_df = train_df.iloc[train_idx], train_df.iloc[val_idx]
        
        # 클래스 불균형 방지
        class_counts = tr_df['label'].value_counts().sort_index().values
        weights = [1.0 / class_counts[lbl] for lbl in tr_df['label'].values]
        sampler = WeightedRandomSampler(weights, num_samples=len(tr_df), replacement=True)
            
        train_loader = DataLoader(PSASiameseDataset(tr_df, 'train'), batch_size=CFG['batch_size'], sampler=sampler, num_workers=CFG['num_workers'], drop_last=True)
        val_loader   = DataLoader(PSASiameseDataset(vl_df, 'val'), batch_size=CFG['batch_size'], shuffle=False, num_workers=CFG['num_workers'])
            
        model = PSASiameseModel_PRO().to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-2)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        
        early_stopping = EarlyStopping(patience=5, path=f'{RES_DIR}/best_model_fold{fold}.pth')
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
            
            history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
            history['val_loss'].append(vl_loss); history['val_acc'].append(vl_acc)
            print(f'  Ep {epoch:02d} | Tr_Loss: {tr_loss:.4f} Tr_Acc: {tr_acc:.4f} | Val_Loss: {vl_loss:.4f} Val_Acc: {vl_acc:.4f}')
            
            # 조기 종료 체크
            early_stopping(vl_loss, model)
            if early_stopping.early_stop:
                print(f"🛑 [조기 종료 발동] 최고 성능 달성 후 더 이상 개선되지 않아 훈련을 조기 중단합니다.")
                break
        
        # 훈련 그래프 저장
        epochs_range = range(1, len(history['train_loss']) + 1)
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1); plt.plot(epochs_range, history['train_loss'], label='Train Loss'); plt.plot(epochs_range, history['val_loss'], label='Val Loss'); plt.legend()
        plt.subplot(1, 2, 2); plt.plot(epochs_range, history['train_acc'], label='Train Acc'); plt.plot(epochs_range, history['val_acc'], label='Val Acc'); plt.legend()
        plt.tight_layout(); plt.savefig(f'{RES_DIR}/Fold_{fold}_Training_Curves.png', dpi=200); plt.close()
        
        # 메모리 정리
        del model, optimizer; gc.collect(); torch.cuda.empty_cache()

    # ==========================================
    # 5. 실전 테스트셋 자동 앙상블 평가
    # ==========================================
    print("\n🏆 [Phase 2] 실전 테스트 153장 앙상블 평가 시작 🏆")
    test_loader = DataLoader(PSASiameseDataset(test_df, 'val'), batch_size=8, shuffle=False, num_workers=4)
    
    models = []
    for fold in range(1, CFG['num_folds'] + 1):
        m = PSASiameseModel_PRO().to(DEVICE)
        m.load_state_dict(torch.load(f'{RES_DIR}/best_model_fold{fold}.pth'))
        m.eval()
        models.append(m)

    def apply_tta(t, i): return t if i == 0 else torch.flip(t, [-1]) if i == 1 else torch.flip(t, [-2])

    all_probs, all_labels = [], []
    with torch.no_grad():
        for f, b, l in test_loader:
            f, b = f.to(DEVICE), b.to(DEVICE)
            probs_list = [F.softmax(m(apply_tta(f, a), apply_tta(b, a)), dim=1) for m in models for a in range(3)]
            all_probs.extend(torch.stack(probs_list).mean(0).cpu().tolist())
            all_labels.extend(l.tolist())

    all_probs = np.array(all_probs)
    preds = np.argmax(all_probs, axis=1)
    target_probs = all_probs[:, 1]
    
    report_txt = classification_report(all_labels, preds, target_names=CLASS_NAMES, digits=4)
    with open(f'{RES_DIR}/FINAL_ULTIMATE_PRO_3_REPORT.txt', 'w', encoding='utf-8') as f: f.write(report_txt)
    print("\n🎯 [ 최종 성적표 ]\n", report_txt)

    cm = confusion_matrix(all_labels, preds)
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1); plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues); plt.colorbar()
    for i, j in np.ndindex(cm.shape): plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > cm.max()/2. else "black")
    fpr, tpr, _ = roc_curve(all_labels, target_probs)
    plt.subplot(1, 3, 2); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc(fpr, tpr):.4f}'); plt.plot([0, 1], linestyle='--'); plt.legend()
    precision, recall, _ = precision_recall_curve(all_labels, target_probs)
    plt.subplot(1, 3, 3); plt.plot(recall, precision, color='green', lw=2, label=f'PR AUC = {auc(recall, precision):.4f}'); plt.legend()
    plt.tight_layout(); plt.savefig(f'{RES_DIR}/FINAL_TEST_PLOTS.png', dpi=150); plt.close()
    
    print(f"✨ 모든 학습 및 평가 완료! 결과 저장 경로: {RES_DIR}")
