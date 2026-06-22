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
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
import torchvision.transforms as T
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# ==========================================
# 1. 환경 설정 (Ultimate Setting)
# ==========================================
RES_DIR = '/data/EunJi/h22000561_psa/v_ultimate'
TRAIN_ROOT = '/data3/home/h22000561/psa_grading/data/raw'
TEST_ROOT = '/data3/home/h22000561/psa_grading/data/test'

os.makedirs(RES_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'>>> Running The Ultimate SOTA (End-to-End Siamese) on: {DEVICE} <<<')

# 모든 크롭 로직 폐기, 512 고해상도로 정면 돌파
CFG = {
    'img_size'    : 512,  
    'batch_size'  : 8,    # 512 해상도 + ConvNeXt이므로 OOM 방지를 위해 8로 설정
    'epochs'      : 25,   
    'num_folds'   : 5,         
    'lr'          : 1e-4,      
    'num_workers' : 4,         
}
CLASS_NAMES = ['Non-Gem (8,9)', 'Gem Mint (10)']

# ==========================================
# 2. 통합 데이터 파이프라인
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
print(f"✅ 학습 데이터: {len(train_df)}장 | 실전 테스트 데이터: {len(test_df)}장 로드 완료")

# 어설픈 크롭 없이 원본 그대로 리사이즈 및 규제
def make_tf(mode='train'):
    norm = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    if mode == 'train':
        return T.Compose([
            T.Resize((CFG['img_size'], CFG['img_size'])),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(), norm,
            T.RandomErasing(p=0.2, scale=(0.02, 0.05)) # 미세 결함 탐지력 극대화
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
# 3. 궁극의 아키텍처: ConvNeXt Siamese Network
# ==========================================
class PSASiameseModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 현존 최강의 뼈대: ConvNeXt Small (특징 추출기로만 사용, num_classes=0)
        self.backbone = timm.create_model('convnext_small', pretrained=True, num_classes=0)
        backbone_out_dim = self.backbone.num_features # 768
        
        # 앞면(768) + 뒷면(768)을 합친 1536 차원을 분석하는 최종 두뇌
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(backbone_out_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, front, back):
        # 샴 네트워크 (하나의 뇌가 앞/뒤를 번갈아 관찰)
        feat_f = self.backbone(front)
        feat_b = self.backbone(back)
        
        # 관찰 결과를 하나로 합침
        fused = torch.cat([feat_f, feat_b], dim=1)
        return self.classifier(fused)

# ==========================================
# 4. 원스톱 학습 및 테스트 루프
# ==========================================
if __name__ == '__main__':
    skf = StratifiedKFold(n_splits=CFG['num_folds'], shuffle=True, random_state=42)
    test_loader = DataLoader(PSASiameseDataset(test_df, 'val'), batch_size=1, shuffle=False, num_workers=4)
    
    print("\n" + "="*50)
    print("🔥 [Phase 1] 5-Fold ConvNeXt Siamese 학습 시작 🔥")
    print("="*50)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label']), 1):
        print(f'\n--- Training Fold {fold}/{CFG["num_folds"]} ---')
        tr_df, vl_df = train_df.iloc[train_idx], train_df.iloc[val_idx]
        
        # 클래스 불균형 해결을 위한 Weighted Sampler
        class_counts = tr_df['label'].value_counts().sort_index().values
        sampler = WeightedRandomSampler(weights=[(1.0 / class_counts)[lbl] for lbl in tr_df['label'].values], num_samples=len(tr_df), replacement=True)
            
        train_loader = DataLoader(PSASiameseDataset(tr_df, 'train'), batch_size=CFG['batch_size'], sampler=sampler, num_workers=4, drop_last=True)
        val_loader   = DataLoader(PSASiameseDataset(vl_df, 'val'), batch_size=CFG['batch_size'], shuffle=False, num_workers=4)
            
        model = PSASiameseModel().to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG['epochs'], eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        
        best_acc = 0.0
        for epoch in range(1, CFG['epochs'] + 1):
            model.train()
            tr_loss = tr_acc = 0
            for f, b, l in train_loader:
                f, b, l = f.to(DEVICE), b.to(DEVICE), l.to(DEVICE)
                optimizer.zero_grad()
                out = model(f, b)
                loss = criterion(out, l)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item() * len(l); tr_acc += (out.argmax(1) == l).sum().item()
            
            scheduler.step()
            tr_loss /= len(tr_df); tr_acc /= len(tr_df)
            
            model.eval()
            vl_loss = vl_acc = 0
            with torch.no_grad():
                for f, b, l in val_loader:
                    f, b, l = f.to(DEVICE), b.to(DEVICE), l.to(DEVICE)
                    out = model(f, b)
                    vl_loss += criterion(out, l).item() * len(l); vl_acc += (out.argmax(1) == l).sum().item()
            vl_loss /= len(vl_df); vl_acc /= len(vl_df)
            
            if vl_acc > best_acc:
                best_acc = vl_acc
                torch.save(model.state_dict(), f'{RES_DIR}/best_model_fold{fold}.pth')
            print(f'  Ep {epoch:02d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}')
            
        del model, optimizer; gc.collect(); torch.cuda.empty_cache()

    print("\n" + "="*50)
    print("🏆 [Phase 2] 실전 테스트셋(Test Set) 자동 앙상블 평가 🏆")
    print("="*50)

    models = []
    for fold in range(1, CFG['num_folds'] + 1):
        m = PSASiameseModel().to(DEVICE)
        m.load_state_dict(torch.load(f'{RES_DIR}/best_model_fold{fold}.pth'))
        m.eval()
        models.append(m)

    def apply_tta(t, i): return t if i == 0 else torch.flip(t, [-1]) if i == 1 else torch.flip(t, [-2])

    all_probs, all_labels = [], []
    with torch.no_grad():
        for f, b, l in test_loader:
            f, b = f.to(DEVICE), b.to(DEVICE)
            probs_list = [F.softmax(m(apply_tta(f, a), apply_tta(b, a)), dim=1) for m in models for a in range(3)]
            all_probs.append(torch.stack(probs_list).mean(0).cpu().numpy()[0])
            all_labels.append(l.item())

    all_probs = np.array(all_probs)
    preds = np.argmax(all_probs, axis=1)
    
    report_txt = classification_report(all_labels, preds, target_names=CLASS_NAMES, digits=4)
    with open(f'{RES_DIR}/FINAL_ULTIMATE_REPORT.txt', 'w', encoding='utf-8') as f_out: f_out.write(report_txt)
    
    print("\n🎯 [ 궁극의 결론 ] 실전 테스트 성적표 🎯")
    print(report_txt)
