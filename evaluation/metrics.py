import os
from torchmetrics.image.kid import KernelInceptionDistance

from numpy import trace, iscomplexobj, cov

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from scipy.linalg import sqrtm
from evaluation.res_net.resnet3D import resnet50
from torch.utils.data import DataLoader, SubsetRandomSampler
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
        if self.transform:
            image = self.transform(image)
        return image


def calculate_activation_statistics(model, dataloader):
    model.eval()
    features = []
    for batch in dataloader:
        batch = batch.unsqueeze(1)
        with torch.no_grad():
            pred = model(batch.float()).cpu().numpy()

            features.append(pred)
    features = np.concatenate(features, axis=0)
    return features


# calculate frechet inception distance
def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def calculate_inception_score(p_yx, eps=1E-16):
    # calculate p(y)
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    # undo the logs
    is_score = np.exp(avg_kl_d)
    return is_score


def calculate_kernel_distance_score(act1, act2):
    kid = KernelInceptionDistance()

    kid.update(act1, real=True)
    kid.update(act2, real=False)
    kid_mean, kid_std = kid.compute()
    return kid_mean, kid_std


def evaluate(real_images_path, generated_images_path, limit=None):
    # Load real samples
    dataset = CustomDataset(real_images_path, transform=transforms.ToTensor())
    if limit is not None:
        real_sampler = SubsetRandomSampler(range(limit))
        real_loader = DataLoader(dataset, batch_size=32, sampler=real_sampler)
    else:
        real_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Load generated samples
    dataset = CustomDataset(generated_images_path, transform=transforms.ToTensor())
    if limit is not None:
        fake_sampler = SubsetRandomSampler(range(limit))
        fake_loader = DataLoader(dataset, batch_size=32, sampler=fake_sampler)
    else:
        fake_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_feature_extractor().to(device)

    # print(f'Calculating activations from real images [{real_images_path}]...')
    act1 = calculate_activation_statistics(model, real_loader)
    # print(f'Calculating activations from fake images [{generated_images_path}]...')
    act2 = calculate_activation_statistics(model, fake_loader)

    fid_score = calculate_fid(act1, act2)
    is_score = calculate_inception_score(act2)

    # kid_mean_score, kid_std_score = calculate_kernel_distance_score(act1, act2)
    # print('FID:', round(fid_score, 5))
    # print('IS:', round(is_score, 5))
    return round(fid_score, 5), round(is_score, 5)
