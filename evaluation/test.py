import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from scipy.linalg import sqrtm

from evaluation.res_net.resnet3D import resnet50
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import time
from argparse import ArgumentParser
from collections import OrderedDict
from torchvision import transforms
from torch.utils.data import Dataset

import numpy as np
from scipy import linalg
import nibabel as nib
import torch
import torch.nn as nn
from torch.nn import functional as F


class Flatten(torch.nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)


def trim_state_dict_name(ckpt):
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_feature_extractor():
    model = resnet50(shortcut_type='B')
    model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                   Flatten())  # (N, 512)
    # ckpt from https://drive.google.com/file/d/1399AsrYpQDi1vq6ciKRQkfknLsQQyigM/view?usp=sharing
    ckpt = torch.load("res_net/weights.pth")
    ckpt = trim_state_dict_name(ckpt["state_dict"])
    model.load_state_dict(ckpt)  # No conv_seg module in ckpt
    model = nn.DataParallel(model).cuda()
    model.eval()
    print("Feature extractor weights loaded")
    return model


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filenames = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_filenames[idx])

        image = nib.load(img_name).get_fdata()
        # if self.transform:
        #     image = self.transform(image)
        return image


def calculate_activation_statistics(model, dataloader, device):
    model.eval()
    features = []
    for batch in dataloader:
        # Add a singleton dimension to the input tensor to represent depth
        # batch = batch.unsqueeze(2).to(device)
        batch = batch.unsqueeze(0)

        print(batch.shape)
        with torch.no_grad():
            features.append(model(batch.float()).cpu().numpy())
    features = np.concatenate(features, axis=0)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen):
    eps = 1e-6
    diff = mu_real - mu_gen
    covmean, _ = sqrtm(sigma_real.dot(sigma_gen), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = sqrtm((sigma_real + offset).dot(sigma_gen + offset))
    fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * np.trace(covmean)
    return fid


dataset = CustomDataset('data/real', transform=transforms.ToTensor())
real_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

dataset = CustomDataset('data/fake', transform=transforms.ToTensor())
fake_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_feature_extractor().to(device)


mu_real, sigma_real = calculate_activation_statistics(model, real_loader, device)
# mu_gen, sigma_gen = calculate_activation_statistics(model, fake_loader, device)

# fid = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
# print("FID:", fid)
