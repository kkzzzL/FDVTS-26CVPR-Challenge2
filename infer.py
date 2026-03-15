import os
import argparse
import torch
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import sys
from utils import *
from tqdm import tqdm
from dataset2 import Lung3D_eccv_patient_supcon
from torch.utils.data import DataLoader
import torch.nn.functional as F
from visualize import Visualizer
from torchnet import meter
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from models.ResNet import SupConResNet
#import segmentation_models_pytorch as smp
#from efficientnet_pytorch_3d import EfficientNet3D
from torch.utils.data import WeightedRandomSampler
#from timm.scheduler.cosine_lr import CosineLRScheduler

import torch.backends.cudnn as cudnn
import random
import math

from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_dataset = Lung3D_eccv_patient_supcon(
    inference=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


model = SupConResNet(name='resnest50_3D', ipt_dim=1, head='mlp', feat_dim=128, n_classes=2, supcon=False)
model.encoder.fc = nn.Linear(2048,4)
model = model.to(device)
model_path = "/remote-home/kejinlu/CVPR2026/challenge2/code/run/checkpoints/con/2026/supcon_mixup_mosmed_bs8_1e-4_resnest50_3D_oversampling/41.pkl"
ckpt = torch.load(model_path, weights_only=False, map_location="cpu")
state_dict = ckpt["net"]
new_state_dict = {}

for k, v in state_dict.items():
    new_key = k.replace("module.", "")
    new_state_dict[new_key] = v
clean_dict = {}

for k, v in new_state_dict.items():
    if k.startswith("head"):
        continue
    if k.startswith("sigma"):
        continue
    clean_dict[k] = v

model.load_state_dict(clean_dict)
model.eval()

results = []
types = ['A', 'G', 'normal', 'covid']
with torch.no_grad():
    for imgs, IDs in tqdm(test_loader, desc="Inference"):

        imgs = imgs.to(device, non_blocking=True)
        _, _, outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        for i in range(len(IDs)):
            results.append([IDs[i], types[preds[i].item()]])

df = pd.DataFrame(results, columns=["id","label"])
df.to_csv("res.csv", index=False)