import os, sys, re, gc
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import timm
import torchvision.transforms as T
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

VERSION = "v_ultimate_pro_4_transparent"
RES_DIR = f'/data/EunJi/h22000561_psa/{VERSION}'
TRAIN_ROOT = '/data3/home/h22000561/psa_grading/data/train_yolo'
TEST_ROOT = '/data3/home/h22000561/psa_grading/data/test_yolo'

os.makedirs(RES_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CFG = {'img_size': 512, 'batch_size': 16, 'accum_steps': 2, 'epochs': 25, 'num_folds': 5, 'lr': 1e-4}
CLASS_NAMES = ['Non-Gem (8,9)', 'Gem Mint (10)']

print("=======================================================")
print(f"🚀 Running {VERSION} (학습 성적표 투명 공개 모드)")
print("=======================================================")

class EarlyStopping:
    def __init__(self, patience=5, path='best_model.pth'):
        self.patience, self.counter, self.best_score, self.early_stop, self.path = patience, 0, None, False, path
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            torch.save(model.state_dict(), self.path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True

def build_robust_dataframe(data_root):
    records = []
    for grade in [8, 9, 10]:
        folder = next((Path(data_root) / f for f in [f'PSA{grade}', f'psa_{grade}', f'psa{grade}'] if (Path(data_root) / f).exists()), None)
        if not folder: continue
        cert_dict = defaultdict(dict)
        for img_path in folder.glob('*.jpg'):
            match = re.search(r'(cert\d+)', img_path.stem.lower())
            if match:
                cid = match.group(1)
                cert_dict[cid]['front' if 'front' in img_path.stem.lower() else 'back'] = str(img_path)
        for sides in cert_dict.values():
            if 'front' in sides and 'back' in sides:
                records.append({'front': sides['front'], 'back': sides['back'], 'label': 1 if grade == 10 else 0})
    return pd.DataFrame(records)

train_df, test_df = build_robust_dataframe(TRAIN_ROOT), build_robust_dataframe(TEST_ROOT)

def make_tf(mode='train'):
    norm = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    if mode == 'train':
        return T.Compose([T.Resize((CFG['img_size'], CFG['img_size'])), T.RandomAffine(3, (0.02, 0.02), (0.98, 1.02)), T.ColorJitter(0.1, 0.1, 0.05), T.RandomHorizontalFlip(), T.ToTensor(), norm])
    return T.Compose([T.Resize((CFG['img_size'], CFG['img_size'])), T.ToTensor(), norm])

class PSASiameseDataset(Dataset):
    def __init__(self, df, mode='train'): self.df, self.tf = df.reset_index(drop=True), make_tf(mode)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx): return self.tf(Image.open(self.df.iloc[idx]['front']).convert('RGB')), self.tf(Image.open(self.df.iloc[idx]['back']).convert('RGB')), self.df.iloc[idx]['label']

class SpatialAttention(nn.Module):
    def __init__(self, dim): super().__init__(); self.conv = nn.Conv2d(dim, 1, 1)
    def forward(self, x): return x * torch.sigmoid(self.conv(x))

class PSASiameseModel_PRO(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = timm.create_model('convnext_small', pretrained=True, num_classes=0, global_pool='')
        dim = self.bb.num_features
        self.attn_f, self.attn_b, self.pool = SpatialAttention(dim), SpatialAttention(dim), nn.AdaptiveAvgPool2d(1)
        # 너무 멍청해지지 않도록 드롭아웃 완화
        self.clf = nn.Sequential(nn.Dropout(0.2), nn.Linear(dim*2, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 2))
    def forward(self, f, b): return self.clf(torch.cat([self.pool(self.attn_f(self.bb(f))).flatten(1), self.pool(self.attn_b(self.bb(b))).flatten(1)], 1))

if __name__ == '__main__':
    skf = StratifiedKFold(CFG['num_folds'], shuffle=True, random_state=42)
    scaler = torch.cuda.amp.GradScaler()
    
    # 훈련 데이터의 클래스 비율 계산 후 파이토치 기본 손실함수에 가중치로 부여 (안전한 밸런싱)
    class_weights = torch.tensor([1.0, len(train_df[train_df['label']==0]) / len(train_df[train_df['label']==1])], dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    all_fold_reports = []
    
    for fold, (t_idx, v_idx) in enumerate(skf.split(train_df, train_df['label']), 1):
        tr_df, vl_df = train_df.iloc[t_idx], train_df.iloc[v_idx]
        tr_ldr = DataLoader(PSASiameseDataset(tr_df, 'train'), batch_size=CFG['batch_size'], shuffle=True, num_workers=8)
        vl_ldr = DataLoader(PSASiameseDataset(vl_df, 'val'), batch_size=CFG['batch_size'], shuffle=False, num_workers=8)
        
        m = PSASiameseModel_PRO().to(DEVICE)
        opt = AdamW(m.parameters(), lr=CFG['lr'], weight_decay=1e-3)
        sch = CosineAnnealingWarmRestarts(opt, T_0=5)
        es = EarlyStopping(5, f'{RES_DIR}/best_f{fold}.pth')
        
        print(f"\n--- Fold {fold} 학습 시작 ---")
        for ep in range(1, CFG['epochs']+1):
            m.train(); tr_loss = 0
            for i, (f, b, l) in enumerate(tr_ldr):
                f, b, l = f.to(DEVICE), b.to(DEVICE), l.to(DEVICE)
                with torch.cuda.amp.autocast(): loss = criterion(m(f, b), l) / CFG['accum_steps']
                scaler.scale(loss).backward()
                if (i+1)%CFG['accum_steps']==0 or (i+1)==len(tr_ldr): scaler.step(opt); scaler.update(); opt.zero_grad()
                tr_loss += loss.item() * CFG['accum_steps']
            sch.step()
            
            m.eval(); vl_loss = 0
            with torch.no_grad():
                for f, b, l in vl_ldr:
                    with torch.cuda.amp.autocast(): vl_loss += criterion(m(f.to(DEVICE), b.to(DEVICE)), l.to(DEVICE)).item()
            
            print(f" Ep {ep:02d} | Tr Loss: {tr_loss/len(tr_ldr):.4f} | Vl Loss: {vl_loss/len(vl_ldr):.4f}")
            es(vl_loss, m)
            if es.early_stop: break

        # ==========================================================
        # [연구자님 요청 사항] 각 Fold 학습 종료 후 "모의고사(Validation)" 성적표 발급
        # ==========================================================
        m.load_state_dict(torch.load(f'{RES_DIR}/best_f{fold}.pth'))
        m.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for f, b, l in vl_ldr:
                val_preds.extend(m(f.to(DEVICE), b.to(DEVICE)).argmax(1).cpu().tolist())
                val_labels.extend(l.tolist())
        
        val_rep = classification_report(val_labels, val_preds, target_names=CLASS_NAMES, digits=4)
        print(f"\n🎯 [Fold {fold} 모의고사(Validation) 상세 성적표] 🎯\n{val_rep}")
        all_fold_reports.append(f"=== Fold {fold} Validation Report ===\n{val_rep}\n")
        
        del m, opt; gc.collect(); torch.cuda.empty_cache()

    # 모의고사 성적표 파일로 통합 저장
    with open(f'{RES_DIR}/TRAINING_VALIDATION_REPORTS.txt', 'w') as f: f.write("\n".join(all_fold_reports))

    # ==========================================================
    # 실전 153장 테스트 앙상블 평가 (기존과 동일)
    # ==========================================================
    print("\n🏆 실전 테스트 153장 앙상블 평가 🏆")
    test_loader = DataLoader(PSASiameseDataset(test_df, 'val'), batch_size=8, shuffle=False)
    models = [PSASiameseModel_PRO().to(DEVICE).eval() for _ in range(CFG['num_folds'])]
    for fold, mod in enumerate(models, 1): mod.load_state_dict(torch.load(f'{RES_DIR}/best_f{fold}.pth'))

    preds, labels = [], []
    with torch.no_grad():
        for f, b, l in test_loader:
            f, b = f.to(DEVICE), b.to(DEVICE)
            probs = torch.stack([torch.nn.functional.softmax(mod(f, b), dim=1) for mod in models]).mean(0)
            preds.extend(probs.argmax(1).cpu().tolist()); labels.extend(l.tolist())
            
    final_rep = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)
    with open(f'{RES_DIR}/FINAL_REAL_TEST_REPORT.txt', 'w') as f: f.write(final_rep)
    print("\n🎯 [ 최종 실전 성적표 ]\n", final_rep)
