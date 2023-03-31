import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.inference.generate_image import generate_image
from visualization.display_image import display_image
import nibabel as nib


def generate_images(experiment, n_images, png, nifti):
    print(experiment)
    model_path = f'saved_models/{experiment}/saved_model'
    _ = generate_image(model_path=model_path)

    out_folder = f'data/generated_images/{experiment}'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if not os.path.exists(f'{out_folder}/png') and png:
        os.makedirs(f'{out_folder}/png')

    if not os.path.exists(f'{out_folder}/nifti') and nifti:
        os.makedirs(f'{out_folder}/nifti')

    print('Generating images...')
    for x in tqdm(range(0, n_images)):
        gen_image = generate_image(model_path=model_path)
        if png:
            fig = display_image(gen_image, show=False, return_figure=True)
            fig.savefig(f'{out_folder}/png/image_{x + 1}.png')
            plt.close()
        if nifti:
            nifti = nib.Nifti1Image(gen_image, np.eye(4))
            nib.save(nifti, f'{out_folder}/nifti/image_{x + 1}.nii.gz')


if __name__ == '__main__':
    if not os.path.exists('data/generated_images'):
        os.makedirs('data/generated_images')
    images = 100
    png = True
    nifti = True

    folders = [f for f in os.listdir('saved_models') if os.path.isdir(os.path.join('saved_models', f))]
    folder_names = [f for f in folders]
    for experiment in tqdm(folder_names):
        if experiment != 'HPO_run_4':
            continue
        try:
            generate_images(experiment=experiment, n_images=images, png=png, nifti=nifti)
        except Exception as e:
            print(f'Failed for {experiment}')
            print(e)
            continue
