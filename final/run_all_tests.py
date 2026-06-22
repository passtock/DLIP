import os
import re
import gc
import traceback
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import torchvision.transforms as T
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# ==========================================
# 1. 설정 및 테스트 경로
# ==========================================
BASE_DIR = '/data/EunJi/h22000561_psa'
TEST_DATA_ROOT = '/data3/home/h22000561/psa_grading/data/test' 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CFG = {'img_size': 300, 'corner_size': 96, 'edge_size': 32}
CLASS_NAMES = ['Non-Gem (8,9)', 'Gem Mint (10)']

print(f'>>> Running ALL 4 Versions (psa_raw_final, v17, v18, v19) Test Inference on: {DEVICE} <<<')

# ==========================================
# 2. 데이터 로더 (디버깅 정보 강화 및 이중 크롭 제거)
# ==========================================
def build_test_dataframe(data_root):
    records = []
    data_root = Path(data_root)
    print(f"\n🔍 테스트 데이터셋 탐색 시작: {data_root}")
    
    for grade in [8, 9, 10]:
        folder_candidates = [data_root / f'PSA{grade}', data_root / f'psa_{grade}', data_root / f'psa{grade}']
        folder = next((f for f in folder_candidates if f.exists()), None)
        
        if not folder:
            print(f"  ⚠️ PSA{grade} 등급에 해당하는 폴더를 찾지 못했습니다. (후보군 확인 바람)")
            continue
            
        img_count = 0
        cert_dict = defaultdict(dict)
        for img_path in folder.glob('*.jpg'):
            img_count += 1
            fn = img_path.stem.lower()
            match = re.search(r'(cert\d+)', fn)
            if match:
                cid = match.group(1)
                if 'front' in fn: cert_dict[cid]['front'] = str(img_path)
                elif 'back' in fn: cert_dict[cid]['back'] = str(img_path)
                
        print(f"  📂 {folder.name} 폴더 검색 완료: 총 {img_count}개의 이미지 발견")
        
        for cert_id, sides in cert_dict.items():
            f_path, b_path = sides.get('front'), sides.get('back')
            if f_path or b_path:
                records.append({
                    'cert': cert_id, 
                    'front': f_path if f_path else b_path, 
                    'back': b_path if b_path else f_path, 
                    'label': 1 if grade == 10 else 0
                })
                
    df = pd.DataFrame(records)
    print(f"📢 [최종 결과] 앞뒤 세트 정렬 완료된 총 카드 수: {len(df)}장\n")
    return df

def crop_regions(img):
    W, H = img.size
    cs, ew, eh = CFG['corner_size'], CFG['img_size'], CFG['edge_size']
    return {
        'full': img, 'tl': img.crop((0, 0, cs, cs)), 'tr': img.crop((W-cs, 0, W, cs)),
        'bl': img.crop((0, H-cs, cs, H)), 'br': img.crop((W-cs, H-cs, W, H)),
        'top': img.crop(((W-ew)//2, 0, (W+ew)//2, eh)), 'bottom': img.crop(((W-ew)//2, H-eh, (W+ew)//2, H)),
        'left': img.crop((0, (H-ew)//2, eh, (H+ew)//2)), 'right': img.crop((W-eh, (H-ew)//2, W, (H+ew)//2)),
        'surface': img.crop((W//4, H//4, 3*W//4, 3*H//4)),
    }

def make_tf(size): return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

class PSATestDataset(Dataset):
    def __init__(self, target_df):
        self.df = target_df.reset_index(drop=True)
        self.tfs = {'full': make_tf(CFG['img_size']), 'corner': make_tf(CFG['corner_size']), 'edge': make_tf(CFG['edge_size']), 'surface': make_tf(CFG['img_size']//2)}
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(row['front']).convert('RGB') as img_f: front_img = img_f.copy()
        with Image.open(row['back']).convert('RGB') as img_b: back_img = img_b.copy()
        
        cf, cb = crop_regions(front_img), crop_regions(back_img)
        f = torch.cat([self.tfs['full'](cf['full']), self.tfs['full'](cb['full'])], dim=0)
        c = torch.stack([self.tfs['corner'](cf[k]) for k in ['tl','tr','bl','br']] + [self.tfs['corner'](cb[k]) for k in ['tl','tr','bl','br']]).reshape(24, CFG['corner_size'], CFG['corner_size'])
        e = torch.stack([self.tfs['edge'](cf[k]) for k in ['top','bottom','left','right']] + [self.tfs['edge'](cb[k]) for k in ['top','bottom','left','right']]).reshape(24, CFG['edge_size'], CFG['edge_size'])
        s = torch.cat([self.tfs['surface'](cf['surface']), self.tfs['surface'](cb['surface'])], dim=0)
        return f, c, e, s, int(row['label']), row['cert']

# ==========================================
# 3. 모델 아키텍처 구조 정의
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

# --- [1] psa_raw_final ---
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
        attn = ((q @ k.transpose(-2, -1)) * (C ** -0.5)).softmax(dim=-1)
        return self.proj(self.attn_drop(attn) @ v)

class PSAModel_RawFinal(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.centering = RegionEncoder(6, embed_dim)
        self.corner = RegionEncoder(24, embed_dim)
        self.edge = RegionEncoder(24, embed_dim)
        self.surface = RegionEncoder(6, embed_dim)
        self.attention_fusion = AttentionFusion(dim=embed_dim)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim * 4, 128), nn.GELU(), nn.Dropout(0.15), nn.Linear(128, 2))
    def forward(self, f, c, e, s):
        features = torch.stack([self.centering(f), self.corner(c), self.edge(e), self.surface(s)], dim=1)
        return self.classifier(self.attention_fusion(features).reshape(features.size(0), -1))

# --- [2] v17 (MHA) ---
class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, dim=128, num_heads=4):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop, self.proj_drop = nn.Dropout(0.1), nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

class PSAModel_v17(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.centering = RegionEncoder(6, embed_dim)
        self.corner = RegionEncoder(24, embed_dim)
        self.edge = RegionEncoder(24, embed_dim)
        self.surface = RegionEncoder(6, embed_dim)
        self.attention_fusion = MultiHeadAttentionFusion(dim=embed_dim, num_heads=4)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim * 4, 128), nn.GELU(), nn.Dropout(0.15), nn.Linear(128, 2))
    def forward(self, f, c, e, s):
        features = torch.stack([self.centering(f), self.corner(c), self.edge(e), self.surface(s)], dim=1)
        return self.classifier(self.attention_fusion(features).reshape(features.size(0), -1))

# --- [3] v18 (Gated) ---
class GatedFeatureFusion(nn.Module):
    def __init__(self, dim=128, num_branches=4):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * num_branches, dim), nn.GELU(), nn.Linear(dim, num_branches), nn.Sigmoid())
    def forward(self, x):
        return x * self.gate(x.reshape(x.size(0), -1)).unsqueeze(-1)

class PSAModel_v18(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.centering = RegionEncoder(6, embed_dim)
        self.corner = RegionEncoder(24, embed_dim)
        self.edge = RegionEncoder(24, embed_dim)
        self.surface = RegionEncoder(6, embed_dim)
        self.gated_fusion = GatedFeatureFusion(dim=embed_dim, num_branches=4)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim * 4, 128), nn.GELU(), nn.Dropout(0.15), nn.Linear(128, 2))
    def forward(self, f, c, e, s):
        features = torch.stack([self.centering(f), self.corner(c), self.edge(e), self.surface(s)], dim=1)
        return self.classifier(self.gated_fusion(features).reshape(features.size(0), -1))

# --- [4] v19 (Transformer) ---
class RegionTransformerFusion(nn.Module):
    def __init__(self, dim=128, num_heads=4):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, 4, dim))
        self.transformer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=512, dropout=0.1, activation='gelu', batch_first=True)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    def forward(self, x):
        return self.transformer(x + self.pos_embed).reshape(x.size(0), -1)

class PSAModel_v19(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.centering = RegionEncoder(6, embed_dim)
        self.corner = RegionEncoder(24, embed_dim)
        self.edge = RegionEncoder(24, embed_dim)
        self.surface = RegionEncoder(6, embed_dim)
        self.transformer_fusion = RegionTransformerFusion(dim=embed_dim, num_heads=4)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim * 4, 128), nn.GELU(), nn.Dropout(0.15), nn.Linear(128, 2))
    def forward(self, f, c, e, s):
        features = torch.stack([self.centering(f), self.corner(c), self.edge(e), self.surface(s)], dim=1)
        return self.classifier(self.transformer_fusion(features))

# ==========================================
# 4. 루프 평가 실행 (try-except 예외 격리 장착)
# ==========================================
if __name__ == '__main__':
    df = build_test_dataframe(TEST_DATA_ROOT)
    if len(df) == 0:
        print("❌ [오류] 외부 테스트셋 데이터를 찾지 못했습니다. 경로 및 파일 확장자를 확인하세요.")
        exit()
        
    test_loader = DataLoader(PSATestDataset(df), batch_size=1, shuffle=False, num_workers=4)

    test_versions = [
        ('psa_raw_final', PSAModel_RawFinal, BASE_DIR),
        ('v17', PSAModel_v17, f"{BASE_DIR}/v17"),
        ('v18', PSAModel_v18, f"{BASE_DIR}/v18"),
        ('v19', PSAModel_v19, f"{BASE_DIR}/v19")
    ]

    for v_name, ModelClass, target_dir in test_versions:
        print("\n" + "="*60)
        print(f"🚀 [ {v_name} ] 외부 테스트셋 앙상블 평가 시도")
        print("="*60)
        
        # 파일 존재 유무 엄격 체크
        if not os.path.exists(f'{target_dir}/best_model_fold1.pth'):
            print(f"⚠️  [패스] {target_dir} 폴더에 가중치 파일(best_model_fold1.pth)이 없습니다. 다음으로 넘어갑니다.")
            continue
            
        try:
            models = []
            print(f"  -> 5-Fold 가중치 로드 중...")
            for fold in range(1, 6):
                m = ModelClass().to(DEVICE)
                m.load_state_dict(torch.load(f'{target_dir}/best_model_fold{fold}.pth', map_location=DEVICE))
                m.eval()
                models.append(m)

            def apply_tta(t, i): return t if i == 0 else torch.flip(t, [-1]) if i == 1 else torch.flip(t, [-2])

            all_probs, all_labels = [], []
            with torch.no_grad():
                for idx, (f, c, e, s, l, cert) in enumerate(test_loader, 1):
                    f, c, e, s = f.to(DEVICE), c.to(DEVICE), e.to(DEVICE), s.to(DEVICE)
                    probs_list = [F.softmax(m(apply_tta(f, a), apply_tta(c, a), apply_tta(e, a), apply_tta(s, a)), dim=1) for m in models for a in range(3)]
                    
                    ensemble_prob = torch.stack(probs_list).mean(0).cpu().numpy()[0]
                    all_probs.append(ensemble_prob)
                    all_labels.append(l.item())
                    
                    if idx % 50 == 0 or idx == len(df):
                        print(f"   [{v_name}] 추론 진행률: {idx}/{len(df)}장 완료")

            all_probs = np.array(all_probs)
            preds_05, target_probs = np.argmax(all_probs, axis=1), all_probs[:, 1]

            report_txt = classification_report(all_labels, preds_05, target_names=CLASS_NAMES, digits=4)
            
            # 파일 쓰기
            out_path = f'{target_dir}/TEST_SET_academic_report_{v_name}.txt'
            with open(out_path, 'w', encoding='utf-8') as f_out:
                f_out.write(f"=== TEST SET Final Report ({v_name}) ===\n")
                f_out.write(report_txt)
            
            print(f"\n✨ [ {v_name} ] 최종 테스트 완료!")
            print(report_txt)

            # 그래프 저장
            cm = confusion_matrix(all_labels, preds_05)
            plt.figure(figsize=(18, 5))
            plt.subplot(1, 3, 1); plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues); plt.colorbar()
            for i, j in np.ndindex(cm.shape): plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > cm.max()/2. else "black")
            fpr, tpr, _ = roc_curve(all_labels, target_probs)
            plt.subplot(1, 3, 2); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Test AUC = {auc(fpr, tpr):.4f}'); plt.plot([0, 1], linestyle='--'); plt.legend()
            precision, recall, _ = precision_recall_curve(all_labels, target_probs)
            plt.subplot(1, 3, 3); plt.plot(recall, precision, color='green', lw=2, label=f'PR AUC = {auc(recall, precision):.4f}'); plt.legend()
            plt.tight_layout(); plt.savefig(f'{target_dir}/TEST_SET_evaluation_plots_{v_name}.png', dpi=150); plt.close()
            
            print(f"🎯 성적표 및 대시보드가 정상 저장되었습니다: {target_dir}/")

        except Exception as e:
            print(f"❌ [에러 패스] {v_name} 평가 중 치명적 요인 발생하여 생략합니다. 이유: {e}")
            traceback.print_exc()
            
        finally:
            if 'models' in locals(): del models
            gc.collect()
            torch.cuda.empty_cache()
