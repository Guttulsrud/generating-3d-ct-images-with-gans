#!/usr/bin/env python

# train HA-GAN
# Hierarchical Amortized GAN for 3D High Resolution Medical Image Synthesis
# https://ieeexplore.ieee.org/abstract/document/9770375
import time

import numpy as np
import torch
import os
import json
import argparse

import yaml
from torch import nn
from torch import optim
from torch.nn import functional as F
from tensorboardX import SummaryWriter
import nibabel as nib
from nilearn import plotting

from utils import trim_state_dict_name, inf_train_gen
from volume_dataset import Volume_Dataset
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True


def main():

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    batch_size = config['batch_size']
    workers = config['workers']
    img_size = config['img_size']
    num_iter = config['iterations']
    log_iter = config['log_iter']
    continue_iter = config['continue_iter']
    latent_dim = config['latent_dim']
    g_iter = config['network']['generator_passes_per_iteration']
    lr_g = config['network']['generator_learning_rate']
    lr_d = config['network']['discriminator_learning_rate']
    lr_e = config['network']['encoder_learning_rate']
    data_dir = config['data_dir']
    exp_name = config['experiment_name']
    fold = config['fold']
    num_class = config['num_class']
    lambda_class = config['lambda_class']

    trainset = Volume_Dataset(data_dir=data_dir, fold=fold, num_class=num_class)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=True,
                                               shuffle=False, num_workers=workers)
    gen_load = inf_train_gen(train_loader)

    if img_size == 256:
        from models.Model_HA_GAN_256 import Discriminator, Generator, Encoder, Sub_Encoder
    elif img_size == 128:
        from models.Model_HA_GAN_128 import Discriminator, Generator, Encoder, Sub_Encoder
    else:
        raise NotImplmentedError

    G = Generator(mode='train', latent_dim=latent_dim, num_class=num_class).cuda()
    D = Discriminator(num_class=num_class).cuda()
    E = Encoder().cuda()
    Sub_E = Sub_Encoder(latent_dim=latent_dim).cuda()

    g_optimizer = optim.Adam(G.parameters(), lr=lr_g, betas=(0.0, 0.999), eps=1e-8)
    d_optimizer = optim.Adam(D.parameters(), lr=lr_d, betas=(0.0, 0.999), eps=1e-8)
    e_optimizer = optim.Adam(E.parameters(), lr=lr_e, betas=(0.0, 0.999), eps=1e-8)
    sub_e_optimizer = optim.Adam(Sub_E.parameters(), lr=lr_e, betas=(0.0, 0.999), eps=1e-8)

    path = f'../logs/{exp_name}/checkpoint'

    # Resume from a previous checkpoint
    if continue_iter != 0:
        ckpt_path = f'{path}/G_iter{str(continue_iter)}.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        ckpt['model'] = trim_state_dict_name(ckpt['model'])
        G.load_state_dict(ckpt['model'])
        g_optimizer.load_state_dict(ckpt['optimizer'])
        ckpt_path = f'{path}/D_iter{str(continue_iter)}.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        ckpt['model'] = trim_state_dict_name(ckpt['model'])
        D.load_state_dict(ckpt['model'])
        d_optimizer.load_state_dict(ckpt['optimizer'])
        ckpt_path = f'{path}/E_iter{str(continue_iter)}.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        ckpt['model'] = trim_state_dict_name(ckpt['model'])
        E.load_state_dict(ckpt['model'])
        e_optimizer.load_state_dict(ckpt['optimizer'])
        ckpt_path = f'{path}/Sub_E_iter{str(continue_iter)}.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        ckpt['model'] = trim_state_dict_name(ckpt['model'])
        Sub_E.load_state_dict(ckpt['model'])
        sub_e_optimizer.load_state_dict(ckpt['optimizer'])
        del ckpt
        print("Ckpt", exp_name, continue_iter, "loaded.")

    G = nn.DataParallel(G)
    D = nn.DataParallel(D)
    E = nn.DataParallel(E)
    Sub_E = nn.DataParallel(Sub_E)

    G.train()
    D.train()
    E.train()
    Sub_E.train()

    real_y = torch.ones((batch_size, 1)).cuda()
    fake_y = torch.zeros((batch_size, 1)).cuda()

    loss_f = nn.BCEWithLogitsLoss()
    loss_mse = nn.L1Loss()

    fake_labels = torch.zeros((batch_size, 1)).cuda()
    real_labels = torch.ones((batch_size, 1)).cuda()

    summary_writer = SummaryWriter(path)

    # # save configurations to a dictionary
    # with open(os.path.join("ha_gan/checkpoint/" + exp_name, 'configs.json'), 'w') as f:
    #     json.dump(vars(args), f, indent=2)

    for p in D.parameters():
        p.requires_grad = False
    for p in G.parameters():
        p.requires_grad = False
    for p in E.parameters():
        p.requires_grad = False
    for p in Sub_E.parameters():
        p.requires_grad = False

    start_time = time.time()
    for iteration in range(continue_iter, num_iter):

        ###############################################
        # Train Discriminator (D^H and D^L)
        ###############################################
        for p in D.parameters():
            p.requires_grad = True
        for p in Sub_E.parameters():
            p.requires_grad = False

        real_images, class_label = gen_load.__next__()
        D.zero_grad()
        real_images = real_images.float().cuda()
        # low-res full volume of real image
        real_images_small = F.interpolate(real_images, scale_factor=0.25)

        # randomly select a high-res sub-volume from real image
        crop_idx = np.random.randint(0, img_size * 7 / 8 + 1)  # 256 * 7/8 + 1
        real_images_crop = real_images[:, :, crop_idx:crop_idx + img_size // 8, :, :]

        if num_class == 0:  # unconditional
            y_real_pred = D(real_images_crop, real_images_small, crop_idx)
            d_real_loss = loss_f(y_real_pred, real_labels)

            # random generation
            noise = torch.randn((batch_size, latent_dim)).cuda()
            # fake_images: high-res sub-volume of generated image
            # fake_images_small: low-res full volume of generated image
            fake_images, fake_images_small = G(noise, crop_idx=crop_idx, class_label=None)
            y_fake_pred = D(fake_images, fake_images_small, crop_idx)

        else:  # conditional
            class_label_onehot = F.one_hot(class_label, num_classes=num_class)
            class_label = class_label.long().cuda()
            class_label_onehot = class_label_onehot.float().cuda()

            y_real_pred, y_real_class = D(real_images_crop, real_images_small, crop_idx)
            # GAN loss + auxiliary classifier loss
            d_real_loss = loss_f(y_real_pred, real_labels) + \
                          F.cross_entropy(y_real_class, class_label)

            # random generation
            noise = torch.randn((batch_size, latent_dim)).cuda()
            fake_images, fake_images_small = G(noise, crop_idx=crop_idx, class_label=class_label_onehot)
            y_fake_pred, y_fake_class = D(fake_images, fake_images_small, crop_idx)

        d_fake_loss = loss_f(y_fake_pred, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()

        d_optimizer.step()

        ###############################################
        # Train Generator (G^A, G^H and G^L)
        ###############################################
        for p in D.parameters():
            p.requires_grad = False
        for p in G.parameters():
            p.requires_grad = True

        for iters in range(g_iter):
            G.zero_grad()

            noise = torch.randn((batch_size, latent_dim)).cuda()
            if num_class == 0:  # unconditional
                fake_images, fake_images_small = G(noise, crop_idx=crop_idx, class_label=None)

                y_fake_g = D(fake_images, fake_images_small, crop_idx)
                g_loss = loss_f(y_fake_g, real_labels)
            else:  # conditional
                fake_images, fake_images_small = G(noise, crop_idx=crop_idx, class_label=class_label_onehot)

                y_fake_g, y_fake_g_class = D(fake_images, fake_images_small, crop_idx)
                g_loss = loss_f(y_fake_g, real_labels) + \
                         lambda_class * F.cross_entropy(y_fake_g_class, class_label)

            g_loss.backward()
            g_optimizer.step()

        ###############################################
        # Train Encoder (E^H)
        ###############################################
        for p in E.parameters():
            p.requires_grad = True
        for p in G.parameters():
            p.requires_grad = False
        E.zero_grad()

        z_hat = E(real_images_crop)
        x_hat = G(z_hat, crop_idx=None)

        e_loss = loss_mse(x_hat, real_images_crop)
        e_loss.backward()
        e_optimizer.step()

        ###############################################
        # Train Sub Encoder (E^G)
        ###############################################
        for p in Sub_E.parameters():
            p.requires_grad = True
        for p in E.parameters():
            p.requires_grad = False
        Sub_E.zero_grad()

        with torch.no_grad():
            z_hat_i_list = []
            # Process all sub-volume and concatenate
            for crop_idx_i in range(0, img_size, img_size // 8):
                real_images_crop_i = real_images[:, :, crop_idx_i:crop_idx_i + img_size // 8, :, :]
                z_hat_i = E(real_images_crop_i)
                z_hat_i_list.append(z_hat_i)
            z_hat = torch.cat(z_hat_i_list, dim=2).detach()
        sub_z_hat = Sub_E(z_hat)
        # Reconstruction
        if num_class == 0:  # unconditional
            sub_x_hat_rec, sub_x_hat_rec_small = G(sub_z_hat, crop_idx=crop_idx)
        else:  # conditional
            sub_x_hat_rec, sub_x_hat_rec_small = G(sub_z_hat, crop_idx=crop_idx, class_label=class_label_onehot)

        sub_e_loss = (loss_mse(sub_x_hat_rec, real_images_crop) + loss_mse(sub_x_hat_rec_small, real_images_small)) / 2.

        sub_e_loss.backward()
        sub_e_optimizer.step()

        # Logging
        if iteration % log_iter == 0:
            summary_writer.add_scalar('D', d_loss.item(), iteration)
            summary_writer.add_scalar('D_real', d_real_loss.item(), iteration)
            summary_writer.add_scalar('D_fake', d_fake_loss.item(), iteration)
            summary_writer.add_scalar('G_fake', g_loss.item(), iteration)
            summary_writer.add_scalar('E', e_loss.item(), iteration)
            summary_writer.add_scalar('Sub_E', sub_e_loss.item(), iteration)

        ###############################################
        # Visualization with Tensorboard
        ###############################################
        if iteration % 200 == 0:
            end_time = time.time()
            runtime_seconds = end_time - start_time
            runtime_minutes, runtime_seconds = divmod(runtime_seconds, 60)
            runtime_hours, runtime_minutes = divmod(runtime_minutes, 60)
            print('[{}/{}]'.format(iteration, num_iter),
                  'D_real: {:<8.3}'.format(d_real_loss.item()),
                  'D_fake: {:<8.3}'.format(d_fake_loss.item()),
                  'G_fake: {:<8.3}'.format(g_loss.item()),
                  'Sub_E: {:<8.3}'.format(sub_e_loss.item()),
                  'E: {:<8.3}'.format(e_loss.item()),
            f"Runtime: {int(runtime_hours)} hours, {int(runtime_minutes)} minutes, {int(runtime_seconds)} seconds")

            featmask = np.squeeze((0.5 * real_images_crop[0] + 0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2, 1, 0)), affine=np.eye(4))
            fig = plt.figure()
            plotting.plot_img(featmask, title="REAL",
                              cut_coords=(img_size // 2, img_size // 2, img_size // 16), figure=fig,
                              draw_cross=False, cmap="gray")
            summary_writer.add_figure('Real', fig, iteration, close=True)

            featmask = np.squeeze((0.5 * sub_x_hat_rec[0] + 0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2, 1, 0)), affine=np.eye(4))
            fig = plt.figure()
            plotting.plot_img(featmask, title="REC",
                              cut_coords=(img_size // 2, img_size // 2, img_size // 16), figure=fig,
                              draw_cross=False, cmap="gray")
            summary_writer.add_figure('Rec', fig, iteration, close=True)

            featmask = np.squeeze((0.5 * fake_images[0] + 0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2, 1, 0)), affine=np.eye(4))
            fig = plt.figure()
            plotting.plot_img(featmask, title="FAKE",
                              cut_coords=(img_size // 2, img_size // 2, img_size // 16), figure=fig,
                              draw_cross=False, cmap="gray")
            summary_writer.add_figure('Fake', fig, iteration, close=True)

        if iteration > 30000 and (iteration + 1) % 500 == 0:
            torch.save({'model': G.state_dict(), 'optimizer': g_optimizer.state_dict()},
                       f'{path}/G_iter{str(iteration + 1)}.pth')
            torch.save({'model': D.state_dict(), 'optimizer': d_optimizer.state_dict()},
                       f'{path}/D_iter{str(iteration + 1)}.pth')
            torch.save({'model': E.state_dict(), 'optimizer': e_optimizer.state_dict()},
                       f'{path}/E_iter{str(iteration + 1)}.pth')
            torch.save({'model': Sub_E.state_dict(), 'optimizer': sub_e_optimizer.state_dict()},
                       f'{path}/Sub_E_iter{str(iteration + 1)}.pth')


if __name__ == '__main__':
    main()
