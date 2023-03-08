import SimpleITK as sitk
import torch
import numpy as np
import matplotlib as mpl

from matplotlib import pyplot as plt
import nibabel as nib

from utils.preprocessing_3d.preprocess_nifti_images import display_image

mpl.use('TkAgg')  # !IMPORTANT

from model.utils import trim_state_dict_name

latent_dim = 1024
save_step = 80000
batch_size = 1
img_size = 128
num_class = 0
exp_name = "saved_model"

if img_size == 256:
    from model.architecture.HA_GAN_256 import Generator, Encoder, Sub_Encoder
elif img_size == 128:
    from model.architecture.HA_GAN_128 import Generator, Encoder, Sub_Encoder

G = Generator(mode='eval', latent_dim=latent_dim, num_class=num_class).cuda()

E = Encoder().cuda()
Sub_E = Sub_Encoder(latent_dim=latent_dim).cuda()

ckpt_path = "../logs/03-07 16-21_concat_128_test/checkpoint" + "/G_iter" + str(save_step) + ".pth"
ckpt = torch.load(ckpt_path)['model']
ckpt = trim_state_dict_name(ckpt)
G.load_state_dict(ckpt)

ckpt_path = "../logs/03-07 16-21_concat_128_test/checkpoint" + "/Eiter" + str(save_step) + ".pth"
ckpt = torch.load(ckpt_path)['model']
ckpt = trim_state_dict_name(ckpt)
E.load_state_dict(ckpt)

ckpt_path = "../logs/03-07 16-21_concat_128_test/checkpoint" + "/Sub_E_iter" + str(save_step) + ".pth"
ckpt = torch.load(ckpt_path)['model']
ckpt = trim_state_dict_name(ckpt)
Sub_E.load_state_dict(ckpt)

print(exp_name, save_step, "step weights loaded.")
del ckpt

G = G.cuda()
E = E.cuda()
Sub_E = Sub_E.cuda()

G.eval()
E.eval()
Sub_E.eval()

torch.cuda.empty_cache()

with torch.no_grad():
    z_rand = torch.randn((batch_size, latent_dim)).cuda()
    x_rand = G(z_rand, 0)
    x_rand = x_rand.detach().cpu().numpy()
    x_rand = 0.5 * x_rand + 0.5  # rescale intensity to [0,1]

    x_rand = x_rand[0, 0, :, :, :]


display_image(x_rand)
exit()
slice_idx = [100, 120, 25]
slice_x = np.flip(x_rand[slice_idx[0], :, :], 0)
slice_y = np.flip(x_rand[:, slice_idx[1], :], 0)
slice_z = np.flip(x_rand[:, :, slice_idx[2]], 0)

result = np.concatenate([slice_x, slice_y], 1)
plt.figure(figsize=(10, 5))
plt.imshow(result, cmap="gray")
plt.axis('off')
plt.show()

#
# low_threshold = -1024
# high_threshold = 600
#
# x_rand_nifti = x_rand * (high_threshold-low_threshold) + low_threshold # rescale to [low_threshold, high_threshold]
# x_rand_nifti = x_rand_nifti.astype(np.int16)
#
# x_rand_nifti = nib.Nifti1Image(x_rand_nifti.transpose((2,1,0)),affine = np.eye(4))
# nib.save(x_rand_nifti, "x_rand_nifti.nii.gz")