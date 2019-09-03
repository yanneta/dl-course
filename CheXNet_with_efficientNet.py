#!/usr/bin/env python
# coding: utf-8

# ## Pneumonia Detection on Chest X-Rays with Deep Learning
# 
# The dataset comes from this [paper](https://arxiv.org/pdf/1711.05225.pdf)
# 
# Also, implementing this [paper] (https://arxiv.org/pdf/1610.02391.pdf)


import numpy as np
import pandas as pd

from pathlib import Path
from sklearn import metrics
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from datetime import datetime
import cv2
from collections import OrderedDict


import os

PATH = Path("/data2/yinterian/ChestXray/")

def read_image(path):
    im = cv2.imread(str(path))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

images_paths = list((PATH/"images").iterdir())


def crop(im, r, c, target_r, target_c): return im[r:r+target_r, c:c+target_c]

def random_crop(x):
    """ Returns a random crop"""
    r, c,*_ = x.shape
    r, c,*_ = x.shape
    r_pix = 8
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    return crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)

def center_crop(x):
    r, c,*_ = x.shape
    r_pix = 8
    c_pix = round(r_pix*c/r)
    return crop(x, r_pix, c_pix, r-2*r_pix, c-2*c_pix)


def rotate_cv(im, deg, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees"""
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, 
                          flags=cv2.WARP_FILL_OUTLIERS+interpolation)


# ## Dataset

# In[8]:


def norm_for_imageNet(img):
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (img - imagenet_stats[0])/imagenet_stats[1]


# In[9]:


def apply_transforms(x):
    """ Applies a random crop, rotation"""
    rdeg = (np.random.random()-.50)*20
    x = rotate_cv(x, rdeg)
    if np.random.random() > 0.5: x = np.fliplr(x).copy() 
    x = random_crop(x)
    return x

class ChestXrayDataSet(Dataset):
    def __init__(self, df, transform=None):
        """
        Args:
            dataframe with data: image_file, label
            transform: if True apply transforms to images
        """
        self.image_files = df["ImageIndex"].values
        self.labels = df["Label"].values
        self.transform = transform
        self.image_path = PATH/"images_250"

    def __getitem__(self, index):
        path = self.image_path/self.image_files[index]
        x = cv2.imread(str(path)).astype(np.float32)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
        if self.transform:
            x = apply_transforms(x)
        else:
            x = center_crop(x)
            
        x = norm_for_imageNet(x)
        y = np.array([int(i) for i in self.labels[index].split(" ")]).astype(np.float32)
        return np.rollaxis(x, 2), y

    def __len__(self):
        return len(self.image_files)


from efficientnet_pytorch import EfficientNet

class EfficientNet0(nn.Module):
    def __init__(self, out_size=14):
        super(EfficientNet0, self).__init__()
        effnet = EfficientNet.from_pretrained('efficientnet-b0')
        layers = list(effnet.children())
        K = len(layers[2])//2
        h1 = layers[:2] + list(layers[2])[:K]
        h2 = list(layers[2])[K:] + layers[3:5]
        self.features1 = nn.Sequential(*h1)
        self.features2 = nn.Sequential(*h2)
        self.classifier = nn.Sequential(*[nn.Linear(1280, out_size)])

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        return self.classifier(x)

class EfficientNet4(nn.Module):
    def __init__(self, out_size=14):
        super(EfficientNet4, self).__init__()
        effnet = EfficientNet.from_pretrained('efficientnet-b4')
        layers = list(effnet.children())
        K = len(layers[2])//2
        h1 = layers[:2] + list(layers[2])[:K]
        h2 = list(layers[2])[K:] + layers[3:5]
        self.features1 = nn.Sequential(*h1)
        self.features2 = nn.Sequential(*h2)
        self.classifier = nn.Sequential(*[nn.Linear(1792, out_size)])

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        return self.classifier(x)


def ave_auc(probs, ys):
    probs = np.vstack(probs)
    ys = np.vstack(ys)
    aucs = [metrics.roc_auc_score(ys[:,i], probs[:,i]) for i in range(probs.shape[1])]
    return np.mean(aucs), aucs


def val_metric(model, valid_dl):
    model.eval()
    total = 0
    sum_loss = 0
    probs = []
    ys = []
    for x, y in valid_dl:
        batch = y.shape[0]
        x = x.cuda().float()
        y = y.cuda().long()
        out = model(x)
        probs.append(out.detach().cpu().numpy())
        ys.append(y.cpu().numpy())
        loss = F.binary_cross_entropy_with_logits(out, y.float())
        sum_loss += batch*(loss.item())
        total += batch
    mean_auc, _ = ave_auc(probs, ys)
    return sum_loss/total, mean_auc


# In[18]:


#val_metric(model, valid_dl)


# ## Training functions

# In[19]:


def cosine_segment(start_lr, end_lr, iterations):
    i = np.arange(iterations)
    c_i = 1 + np.cos(i*np.pi/iterations)
    return end_lr + (start_lr - end_lr)/2 *c_i

def get_cosine_triangular_lr(max_lr, iterations):
    min_start, min_end = max_lr/25, max_lr/(25*1e4)
    iter1 = int(0.3*iterations)
    iter2 = iterations - iter1
    segs = [cosine_segment(min_start, max_lr, iter1), cosine_segment(max_lr, min_end, iter2)]
    return np.concatenate(segs)



def create_optimizer(model, lr0):
    params = [{'params': model.features1.parameters(), 'lr': lr0/9},
              {'params': model.features2.parameters(), 'lr': lr0/3},
              {'params': model.classifier.parameters(), 'lr': lr0}]
    return optim.Adam(params, weight_decay=1e-5)

def update_optimizer(optimizer, group_lrs):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = group_lrs[i]


def train_epoch(model, train_dl, optimizer, lrs, idx):
    model.train()
    total = 0
    sum_loss = 0
    for i, (x, y) in enumerate(train_dl):
        lr = lrs[idx]
        update_optimizer(optimizer, [lr/9, lr/3, lr])
        batch = y.shape[0]
        x = x.cuda().float()
        y = y.cuda().float()
        out = model(x)
        loss = F.binary_cross_entropy_with_logits(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        idx += 1
        total += batch
        sum_loss += batch*(loss.item())
    train_loss = sum_loss/total
    return train_loss, idx  



def train_triangular_policy(model, train_dl, valid_dl, max_lr=0.04, epochs = 5):
    idx = 0
    iterations = epochs*len(train_dl)
    lrs = get_cosine_triangular_lr(max_lr, iterations)
    optimizer = create_optimizer(model, lrs[0])
    prev_val_auc = 0.0
    for i in range(epochs):
        train_loss, idx = train_epoch(model, train_dl, optimizer, lrs, idx)
        val_loss, val_auc = val_metric(model, valid_dl)
        results = "epoch %d train_loss %.3f val_loss %.3f val_auc %.3f \n" % (
                i+1, train_loss, val_loss, val_auc)
        print(results)
        f.write(results)
        f.write('\n')
        f.flush()
        if val_auc > prev_val_auc:
            prev_val_auc = val_auc
            path = "{0}/models/model_efficient_net0_auc_{1:.0f}.pth".format(PATH, 100*val_auc) 
            save_model(model, path)
            print(path)

def save_model(m, p): torch.save(m.state_dict(), p)
    
def load_model(m, p): m.load_state_dict(torch.load(p))

PATH = Path("/data2/yinterian/ChestXray/")


train_df = pd.read_csv(PATH/"train_df.csv")
val_df = pd.read_csv(PATH/"val_df.csv")
test_df = pd.read_csv(PATH/"test_df.csv")

train_ds = ChestXrayDataSet(train_df, transform=True)
valid_ds = ChestXrayDataSet(val_df)
train_dl = DataLoader(train_ds, batch_size=40, shuffle=True, num_workers=1)
valid_dl = DataLoader(valid_ds, batch_size=16, num_workers=1)

f = open('out_eff0_lr_005_ep_15.log', 'w+')
model = EfficientNet0().cuda()
train_triangular_policy(model, train_dl, valid_dl, max_lr=0.005, epochs = 15)
f.close()

f = open('out_eff0_lr_01_ep_15.log', 'w+')
model = EfficientNet0().cuda()
train_triangular_policy(model, train_dl, valid_dl, max_lr=0.005, epochs = 15)
f.close()


