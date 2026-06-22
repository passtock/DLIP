import sys
# 기존 스크립트의 내용을 그대로 가져오되, 백본만 B4로 바꿉니다.
with open('train_v22_ultimate.py', 'r') as f:
    code = f.read()

# B2를 B4로 교체하는 마법
code = code.replace('efficientnet_b2', 'efficientnet_b4')
code = code.replace('EfficientNet_B2_Weights', 'EfficientNet_B4_Weights')
code = code.replace('v22_ultimate', 'v22_b4_upgrade')

with open('train_v22_b4_run.py', 'w') as f:
    f.write(code)
