#!/bin/bash
#SBATCH --job-name=prob_extract
#SBATCH --output=logs/prob_extract_%j.out
#SBATCH --error=logs/prob_extract_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

export LD_LIBRARY_PATH=/home/sonic/anaconda3/lib:$LD_LIBRARY_PATH

cd ~/psa_grading

python << 'PYTHON'
import sklearn
print("SKLEARN OK:", sklearn.__version__)

import importlib
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re
from torch.utils.data import DataLoader

OUTDIR="/data/EunJi/h22000561_psa/prob_analysis"
Path(OUTDIR).mkdir(exist_ok=True)

TEST_ROOT="/data3/home/h22000561/psa_grading/data/test_yolo"

def build_test_df():
    records=[]

    for grade in [8,9,10]:
        folder=Path(TEST_ROOT)/f"psa_{grade}"

        cert_dict=defaultdict(dict)

        for img_path in folder.glob("*.jpg"):
            fn=img_path.stem.lower()

            m=re.search(r'(cert\d+)',fn)

            if m:
                cid=m.group(1)

                if "front" in fn:
                    cert_dict[cid]["front"]=str(img_path)

                elif "back" in fn:
                    cert_dict[cid]["back"]=str(img_path)

        for cid,sides in cert_dict.items():
            if "front" in sides and "back" in sides:
                records.append({
                    "front":sides["front"],
                    "back":sides["back"],
                    "label":1 if grade==10 else 0
                })

    return pd.DataFrame(records)

test_df=build_test_df()

print("TOTAL TEST CARDS =",len(test_df))

for mod_name,resdir in [
    ("v18_2","/data/EunJi/h22000561_psa/v18_2"),
    ("v19_2","/data/EunJi/h22000561_psa/v19_2"),
    ("v20_2","/data/EunJi/h22000561_psa/v20_2")
]:

    print("="*80)
    print(mod_name)
    print("="*80)

    mod=importlib.import_module(mod_name)

    dataset=mod.PSADataset(test_df,"val")

    loader=DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )

    device=torch.device("cuda")

    models=[]

    for fold in range(1,6):

        m=mod.PSAMultiBranchModel().to(device)

        m.load_state_dict(
            torch.load(
                f"{resdir}/best_model_fold{fold}.pth",
                map_location=device
            )
        )

        m.eval()

        models.append(m)

    probs_all=[]
    labels_all=[]

    with torch.no_grad():

        for f,c,e,s,l in loader:

            f=f.to(device)
            c=c.to(device)
            e=e.to(device)
            s=s.to(device)

            fold_probs=[]

            for m in models:
                fold_probs.append(
                    F.softmax(
                        m(f,c,e,s),
                        dim=1
                    )
                )

            probs=torch.stack(fold_probs).mean(0)

            probs_all.extend(
                probs[:,1].cpu().numpy()
            )

            labels_all.extend(
                l.numpy()
            )

    df=pd.DataFrame({
        "true_label":labels_all,
        "prob_gem10":probs_all
    })

    csv_file=f"{OUTDIR}/{mod_name}_test_probs.csv"

    df.to_csv(csv_file,index=False)

    print("SAVED:",csv_file)

    print(
        "Mean Prob 10 =",
        df[df.true_label==1].prob_gem10.mean()
    )

    print(
        "Mean Prob 8/9 =",
        df[df.true_label==0].prob_gem10.mean()
    )

PYTHON
