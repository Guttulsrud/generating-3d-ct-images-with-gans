# Logging
import numpy as np
from matplotlib import pyplot as plt
from nilearn import plotting
import torch
import nibabel as nib


def log_training(config, data, logger):
    start_model_saving = config['start_model_saving']
    save_model_interval = config['save_model_interval']
    tensorboard_log_interval = config['tensorboard_log_interval']
    print_log_interval = config['print_log_interval']
    num_iter = config['num_iter']
    img_size = config['img_size']
    exp_name = config['exp_name']

    iteration = data['iteration']
    d_loss = data['d_loss']
    g_loss = data['g_loss']
    e_loss = data['e_loss']
    sub_e_loss = data['sub_e_loss']
    d_real_loss = data['d_real_loss']
    d_fake_loss = data['d_fake_loss']
    summary_writer = data['summary_writer']
    real_images_crop = data['real_images_crop']
    sub_x_hat_rec = data['sub_x_hat_rec']
    fake_images = data['fake_images']
    G = data['G']
    D = data['D']
    E = data['E']
    Sub_E = data['Sub_E']
    g_optimizer = data['g_optimizer']
    d_optimizer = data['d_optimizer']
    e_optimizer = data['e_optimizer']
    sub_e_optimizer = data['sub_e_optimizer']

    if iteration % tensorboard_log_interval == 0:
        summary_writer.add_scalar('D', d_loss.item(), iteration)
        summary_writer.add_scalar('D_real', d_real_loss.item(), iteration)
        summary_writer.add_scalar('D_fake', d_fake_loss.item(), iteration)
        summary_writer.add_scalar('G_fake', g_loss.item(), iteration)
        summary_writer.add_scalar('E', e_loss.item(), iteration)
        summary_writer.add_scalar('Sub_E', sub_e_loss.item(), iteration)

    ###############################################
    # Visualization with Tensorboard
    ###############################################
    if iteration % print_log_interval == 0:
        print('[{}/{}]'.format(iteration, num_iter),
              'D_real loss: {:<8.3}'.format(d_real_loss.item()),
              'D_fake loss: {:<8.3}'.format(d_fake_loss.item()),
              'G_fake loss: {:<8.3}'.format(g_loss.item()),
              'Sub_E loss: {:<8.3}'.format(sub_e_loss.item()),
              'E loss: {:<8.3}'.format(e_loss.item()))

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

    path = f'{logger.path}/checkpoint/'
    if iteration > start_model_saving and (iteration + 1) % save_model_interval == 0:
        torch.save({'model': G.state_dict(), 'optimizer': g_optimizer.state_dict()},
                   f'{path}/G_iter{str(iteration + 1)}.pth')
        torch.save({'model': D.state_dict(), 'optimizer': d_optimizer.state_dict()},
                   f'{path}/D_iter{str(iteration + 1)}.pth')
        torch.save({'model': E.state_dict(), 'optimizer': e_optimizer.state_dict()},
                   f'{path}/Eiter{str(iteration + 1)}.pth')
        torch.save({'model': Sub_E.state_dict(), 'optimizer': sub_e_optimizer.state_dict()},
                   f'{path}/Sub_E_iter{str(iteration + 1)}.pth')
