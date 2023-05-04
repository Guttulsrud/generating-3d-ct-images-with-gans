import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.inference.generate_image import generate_image
from visualization.display_image import display_image
import nibabel as nib


def generate_images(experiment, n_png, n_nifti, png, nifti, im_size):

    model_path = f'saved_models/{experiment}/saved_model'
    _ = generate_image(model_path=model_path, img_size=im_size)

    out_folder = f'data/generated_images/{experiment}'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if not os.path.exists(f'{out_folder}/png') and png:
        os.makedirs(f'{out_folder}/png')

    if not os.path.exists(f'{out_folder}/nifti') and nifti:
        os.makedirs(f'{out_folder}/nifti')

    generated_png = 0
    for x in range(0, n_nifti):
        gen_image = generate_image(model_path=model_path, img_size=im_size, rescale_intensity=False)
        if png and generated_png < n_png:
            fig = display_image(gen_image, show=False, return_figure=True)
            fig.savefig(f'{out_folder}/png/image_{x + 1}.png')
            plt.close()
            generated_png += 1
        if nifti:
            nifti = nib.Nifti1Image(gen_image, np.eye(4))
            nib.save(nifti, f'{out_folder}/nifti/image_{x + 1}.nii.gz')


if __name__ == '__main__':
    if not os.path.exists('data/generated_images'):
        os.makedirs('data/generated_images')
    n_png = 30
    n_nifti = 100
    png = True
    nifti = False
    latent_dim = 1024
    im_size = 256

    folders = [f for f in os.listdir('saved_models') if os.path.isdir(os.path.join('saved_models', f))]
    folder_names = [f for f in folders]

    models_to_test = [
        # 'Latent_dim1280',
        # 'Latent_dim1536',
        '256_hpo_17_non_norm',
        # 'modified_network',
    ]

    for experiment in tqdm(folder_names):
        if experiment not in models_to_test:
            continue

        if os.path.exists(f'data/generated_images/{experiment}'):
            continue

        try:
            print(experiment)
            generate_images(experiment=experiment, n_png=n_png, n_nifti=n_nifti, png=png, nifti=nifti, im_size=im_size)
        except Exception as e:
            print(f'Failed for {experiment}')
            print(e)
            continue
