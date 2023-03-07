import numpy as np
import torch
import os
import json
import yaml
from torch import nn
from torch import optim
from torch.nn import functional as F
from tensorboardX import SummaryWriter
import matplotlib as mpl
from dataloader.ha_gan_data_loader import get_data_loader
from model.get_model import get_model
from model.logging import log_training
from model.utils import trim_state_dict_name

mpl.use('Agg')
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def continue_checkpoint(name, checkpoint_path, model, optimizer, continue_iter):
    ckpt_path = f'{checkpoint_path}/{name}_iter{str(continue_iter)}.pth'
    ckpt = torch.load(ckpt_path, map_location='cuda')
    ckpt['model'] = trim_state_dict_name(ckpt['model'])
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])


def train(logger, config):
    gen_load = get_data_loader(config)

    G, D, E, Sub_E, = get_model(config)
    generator_learning_rate = config['network']['generator_learning_rate']
    discriminator_learning_rate = config['network']['discriminator_learning_rate']
    encoder_learning_rate = config['network']['encoder_learning_rate']
    continue_iter = config['continue_iter']
    exp_name = config['exp_name']
    num_class = config['num_class']
    latent_dim = config['latent_dim']
    batch_size = config['batch_size']
    img_size = config['img_size']
    num_iter = config['num_iter']
    generator_passes_per_iteration = config['network']['generator_passes_per_iteration']
    lambda_class = config['lambda_class']
    g_optimizer = optim.Adam(G.parameters(), lr=generator_learning_rate, betas=(0.0, 0.999), eps=1e-8)
    d_optimizer = optim.Adam(D.parameters(), lr=discriminator_learning_rate, betas=(0.0, 0.999), eps=1e-8)
    e_optimizer = optim.Adam(E.parameters(), lr=encoder_learning_rate, betas=(0.0, 0.999), eps=1e-8)
    sub_e_optimizer = optim.Adam(Sub_E.parameters(), lr=encoder_learning_rate, betas=(0.0, 0.999), eps=1e-8)

    # Resume from a previous checkpoint

    checkpoint_path = f'{logger.path}'
    if continue_iter != 0:
        continue_checkpoint(name='G',
                            checkpoint_path=checkpoint_path,
                            model=G,
                            optimizer=g_optimizer,
                            continue_iter=continue_iter)
        continue_checkpoint(name='D',
                            checkpoint_path=checkpoint_path,
                            model=D,
                            optimizer=d_optimizer,
                            continue_iter=continue_iter)
        continue_checkpoint(name='E',
                            checkpoint_path=checkpoint_path,
                            model=E,
                            optimizer=e_optimizer,
                            continue_iter=continue_iter)
        continue_checkpoint(name='Sub_E',
                            checkpoint_path=checkpoint_path,
                            model=Sub_E,
                            optimizer=sub_e_optimizer,
                            continue_iter=continue_iter)

        print("Ckpt", exp_name, continue_iter, "loaded.")

    G = nn.DataParallel(G)
    D = nn.DataParallel(D)
    E = nn.DataParallel(E)
    Sub_E = nn.DataParallel(Sub_E)

    G.train()
    D.train()
    E.train()
    Sub_E.train()

    loss_f = nn.BCEWithLogitsLoss()
    loss_mse = nn.L1Loss()

    fake_labels = torch.zeros((batch_size, 1)).cuda()
    real_labels = torch.ones((batch_size, 1)).cuda()

    summary_writer = SummaryWriter(f'{logger.path}/checkpoint')

    for p in D.parameters():
        p.requires_grad = False
    for p in G.parameters():
        p.requires_grad = False
    for p in E.parameters():
        p.requires_grad = False
    for p in Sub_E.parameters():
        p.requires_grad = False

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

        for iters in range(generator_passes_per_iteration):
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
        log_training(config=config,
                     data={
                         'iteration': iteration,
                         'd_loss': d_loss,
                         'g_loss': g_loss,
                         'e_loss': e_loss,
                         'sub_e_loss': sub_e_loss,
                         'd_real_loss': d_real_loss,
                         'd_fake_loss': d_fake_loss,
                         'summary_writer': summary_writer,
                         'real_images_crop': real_images_crop,
                         'sub_x_hat_rec': sub_x_hat_rec,
                         'fake_images': fake_images,
                         'G': G,
                         'D': D,
                         'E': E,
                         'Sub_E': Sub_E,
                         'g_optimizer': g_optimizer,
                         'd_optimizer': d_optimizer,
                         'e_optimizer': e_optimizer,
                         'sub_e_optimizer': sub_e_optimizer}, logger=logger)
