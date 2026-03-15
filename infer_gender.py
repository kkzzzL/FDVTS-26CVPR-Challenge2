import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from dataset2 import Lung3D_eccv_patient_supcon
from models.ResNet import SupConResNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_dataset = Lung3D_eccv_patient_supcon(inference=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

gender_model = SupConResNet(
    name='resnest50_3D',
    ipt_dim=1,
    head='mlp',
    feat_dim=128,
    n_classes=2,
    supcon=False
)

gender_model.encoder.fc = nn.Linear(2048, 2)

gender_ckpt = torch.load("/remote-home/kejinlu/CVPR2026/challenge2/code/run/checkpoints/con/2026/clf_bs6_5e-5_resnest50/50.pkl", weights_only=False, map_location="cpu")
state_dict = gender_ckpt["net"]

new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
gender_model.load_state_dict(new_state_dict, strict=False)

gender_model = gender_model.to(device)
gender_model.eval()

male_model = SupConResNet(
    name='resnest50_3D',
    ipt_dim=1,
    head='mlp',
    feat_dim=128,
    n_classes=2,
    supcon=False
)

male_model.encoder.fc = nn.Linear(2048, 4)

male_ckpt = torch.load("/remote-home/kejinlu/CVPR2026/challenge2/code/run/checkpoints/con/2026/supcon_mixup_mosmed_bs8_1e-4_resnest50_3D_male/38.pkl", weights_only=False, map_location="cpu")
state_dict = male_ckpt["net"]
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
male_model.load_state_dict(new_state_dict, strict=False)

male_model = male_model.to(device)
male_model.eval()

female_model = SupConResNet(
    name='resnest50_3D',
    ipt_dim=1,
    head='mlp',
    feat_dim=128,
    n_classes=2,
    supcon=False
)

female_model.encoder.fc = nn.Linear(2048, 4)

female_ckpt = torch.load("/remote-home/kejinlu/CVPR2026/challenge2/code/run/checkpoints/con/2026/supcon_mixup_mosmed_bs8_1e-4_resnest50_3D_female/54.pkl", weights_only=False, map_location="cpu")
state_dict = female_ckpt["net"]
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
female_model.load_state_dict(new_state_dict, strict=False)

female_model = female_model.to(device)
female_model.eval()

results = []
types = ['A', 'G', 'normal', 'covid']
gender_count = {
    "A": {"male": 0, "female": 0},
    "G": {"male": 0, "female": 0},
    "normal": {"male": 0, "female": 0},
    "covid": {"male": 0, "female": 0},
}

with torch.no_grad():

    for imgs, IDs in tqdm(test_loader, desc="Inference"):

        imgs = imgs.to(device, non_blocking=True)
        gender_logits = gender_model(imgs)

        if isinstance(gender_logits, tuple):
            gender_logits = gender_logits[-1]

        gender_pred = torch.argmax(gender_logits, dim=1)

        male_idx = (gender_pred == 1).nonzero(as_tuple=True)[0]
        female_idx = (gender_pred == 0).nonzero(as_tuple=True)[0]

        batch_preds = torch.zeros(len(imgs), dtype=torch.long, device=device)

        if len(male_idx) > 0:
            male_imgs = imgs[male_idx]
            _, _, outputs = male_model(male_imgs)

            if isinstance(outputs, tuple):
                outputs = outputs[-1]

            preds = torch.argmax(outputs, dim=1)
            batch_preds[male_idx] = preds

        if len(female_idx) > 0:
            female_imgs = imgs[female_idx]
            _, _, outputs = female_model(female_imgs)

            if isinstance(outputs, tuple):
                outputs = outputs[-1]

            preds = torch.argmax(outputs, dim=1)
            batch_preds[female_idx] = preds

        for i in range(len(IDs)):

            label = types[batch_preds[i].item()]
            gender = "male" if gender_pred[i].item() == 1 else "female"

            results.append([IDs[i], label])

            gender_count[label][gender] += 1
            
df = pd.DataFrame(results, columns=["id", "label"])
df.to_csv("hierarchical.csv", index=False)

print("Done.")

print("\n===== Gender Distribution Per Disease =====")
for cls in types:
    print(f"{cls}:")
    print(f"   male   = {gender_count[cls]['male']}")
    print(f"   female = {gender_count[cls]['female']}")

stat_df = pd.DataFrame(gender_count).T
stat_df.to_csv("gender_distribution_per_class.csv")
