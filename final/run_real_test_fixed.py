import os, sys, re, torch
import importlib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

VERSION = sys.argv[1]
# 주의: _2 모델들이므로 반드시 누끼가 따진 test_yolo를 사용해야 합니다.
TEST_ROOT = '/data3/home/h22000561/psa_grading/data/test_yolo' 
RES_DIR = f'/data/EunJi/h22000561_psa/{VERSION}'

# --- [수정됨] 0장 로드 버그를 방지하는 튼튼한 탐색기 ---
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

# 153장이 제대로 로드되는지 확인
test_df = build_robust_dataframe(TEST_ROOT)
print(f"\n=======================================================")
print(f"🚀 [{VERSION}] 실전 테스트셋 로드 완료: 총 {len(test_df)}장")
print(f"=======================================================")

if len(test_df) == 0:
    print("❌ [치명적 에러] 테스트 데이터를 한 장도 찾지 못했습니다!")
    sys.exit(1)

# 모델 구조 빌려오기
mod = importlib.import_module(VERSION)
PSAMultiBranchModel = mod.PSAMultiBranchModel
PSADataset = mod.PSADataset
DEVICE = mod.DEVICE
CLASS_NAMES = mod.CLASS_NAMES

test_dataset = PSADataset(test_df, mode='val')
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

models = []
for fold in range(1, 6):
    m = PSAMultiBranchModel().to(DEVICE)
    weight_path = f'{RES_DIR}/best_model_fold{fold}.pth'
    if not os.path.exists(weight_path):
        print(f"❌ [에러] {weight_path} 가 없습니다. 해당 폴드 학습이 실패했거나 누락되었습니다.")
        sys.exit(1)
    m.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    m.eval()
    models.append(m)

def apply_tta(tensor, tta_idx):
    if tta_idx == 0: return tensor
    elif tta_idx == 1: return torch.flip(tensor, [-1])
    return torch.flip(tensor, [-2])

all_probs, all_labels = [], []
with torch.no_grad():
    for idx, (f, c, e, s, l) in enumerate(test_loader, 1):
        f, c, e, s = f.to(DEVICE), c.to(DEVICE), e.to(DEVICE), s.to(DEVICE)
        probs_list = []
        for m in models:
            for aug_idx in range(3):
                probs_list.append(F.softmax(m(apply_tta(f, aug_idx), apply_tta(c, aug_idx), apply_tta(e, aug_idx), apply_tta(s, aug_idx)), dim=1))
        all_probs.extend(torch.stack(probs_list).mean(0).cpu().tolist())
        all_labels.extend(l.tolist())
        if idx % 10 == 0: print(f"  [{VERSION}] 평가 중... {idx*8}/{len(test_df)}")

all_probs = np.array(all_probs)
preds_05 = np.argmax(all_probs, axis=1)
target_probs = all_probs[:, 1]

report_txt = classification_report(all_labels, preds_05, target_names=CLASS_NAMES, digits=4)
with open(f'{RES_DIR}/REAL_TEST_REPORT.txt', 'w', encoding='utf-8') as f_out:
    f_out.write(f"=== REAL TEST (153 Cards) Report for {VERSION} ===\n")
    f_out.write(report_txt)

cm = confusion_matrix(all_labels, preds_05)
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1); plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues); plt.colorbar()
for i, j in np.ndindex(cm.shape): plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > cm.max()/2. else "black")

fpr, tpr, _ = roc_curve(all_labels, target_probs)
plt.subplot(1, 3, 2); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc(fpr, tpr):.4f}'); plt.plot([0, 1], linestyle='--'); plt.legend()

precision, recall, _ = precision_recall_curve(all_labels, target_probs)
plt.subplot(1, 3, 3); plt.plot(recall, precision, color='green', lw=2, label=f'PR AUC = {auc(recall, precision):.4f}'); plt.legend()

plt.tight_layout(); plt.savefig(f'{RES_DIR}/REAL_TEST_PLOTS.png', dpi=150); plt.close()
