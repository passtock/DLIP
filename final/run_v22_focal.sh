#!/bin/bash
#SBATCH --job-name=v22_Foc
#SBATCH --output=/data/EunJi/h22000561_psa/logs/train_v22_focal_%j.out
#SBATCH --error=/data/EunJi/h22000561_psa/logs/train_v22_focal_%j.err
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1

# 1. 안전한 디렉토리 및 환경 설정
mkdir -p /data/EunJi/h22000561_psa/logs
export LD_LIBRARY_PATH=/home/sonic/anaconda3/lib:$LD_LIBRARY_PATH

# 2. 콘다 환경 강제 활성화
source /home/sonic/anaconda3/etc/profile.d/conda.sh
conda activate h22000561-psa

cd /data3/home/h22000561/psa_grading/

# 3. Focal Loss 파이썬 코드 생성 및 실행
python -c "
with open('train_v22_ultimate.py', 'r') as f:
    code = f.read()

focal_code = '''
class FocalLoss(nn.Module):
    def __init__(self, alpha=[1.5, 1.2, 0.5], gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[targets] * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()
'''
code = code.replace('class BranchEncoder', focal_code + '\nclass BranchEncoder')
code = code.replace('criterion = nn.CrossEntropyLoss(label_smoothing=0.1)', 'criterion = FocalLoss()')
code = code.replace('v22_ultimate', 'v22_focal_loss')

with open('train_v22_focal_run.py', 'w') as f:
    f.write(code)
"

python train_v22_focal_run.py
