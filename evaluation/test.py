import os

from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from scipy.linalg import sqrtm
from evaluation.res_net.resnet3D import resnet50
from torch.utils.data import DataLoader
import os
from collections import OrderedDict
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from scipy import linalg
import sys


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
    model.load_state_dict(ckpt)
    model = nn.DataParallel(model).cuda()
    model.eval()
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


def calculate_activation_statistics(model, dataloader, device, limit=None):
    model.eval()
    features = []
    samples = 0
    for batch in dataloader:
        for idx, el in enumerate(batch):
            if limit is not None and samples >= limit:
                break
            samples += 1
            print(f'Image {idx + 1}/{limit}')

            el = el.unsqueeze(0).to(device)
            el = el.unsqueeze(0).to(device)
            with torch.no_grad():
                features.append(model(el.float()).cpu().numpy())
            if limit is not None and samples >= limit:
                break
        if limit is not None and samples >= limit:
            break
    features = np.concatenate(features, axis=0)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen, eps=1e-6):
    eps = 1e-6
    mu1 = np.atleast_1d(mu_real)
    mu2 = np.atleast_1d(mu_gen)
    sigma1 = np.atleast_2d(sigma_real)
    sigma2 = np.atleast_2d(sigma_gen)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Add a small constant value to the diagonal of the covariance matrices
    sigma1 += eps * np.eye(sigma1.shape[0])
    sigma2 += eps * np.eye(sigma2.shape[0])

    # Compute the matrix square root of the product of the covariance matrices
    try:
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    except ValueError:
        # Use a fallback method if sqrtm fails due to numerical instability
        offset = eps * np.eye(sigma1.shape[0])
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


limit = 100
dataset = CustomDataset('../data/concat', transform=transforms.ToTensor())
real_loader = DataLoader(dataset, batch_size=limit, shuffle=True)

dataset = CustomDataset('../data/generated_images/HPO_run_4/nifti', transform=transforms.ToTensor())
fake_loader = DataLoader(dataset, batch_size=limit, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_feature_extractor().to(device)

print('Calculating real statistics...')
mu_real, sigma_real = calculate_activation_statistics(model, real_loader, device, limit)
print('Calculating generated statistics...')
mu_gen, sigma_gen = calculate_activation_statistics(model, fake_loader, device, limit)

print('Calculating FID...')
fid = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
print("FID:", fid)
