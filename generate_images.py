import os

from postprocessing.cca import connected_component_analysis
from utils.gen_utils import create_out_folders

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.inference.generate_image import generate_image
from visualization.display_image import display_image
import nibabel as nib


def create_folders(out_folder):
    if not os.path.exists(f'{out_folder}/png') and png:
        os.makedirs(f'{out_folder}/png')

    if not os.path.exists(f'{out_folder}/png/raw') and png:
        os.makedirs(f'{out_folder}/png/raw')

    if not os.path.exists(f'{out_folder}/png/post_processed') and png:
        os.makedirs(f'{out_folder}/png/post_processed')

    if not os.path.exists(f'{out_folder}/png/post_processed/images') and png:
        os.makedirs(f'{out_folder}/png/post_processed/images')

    if not os.path.exists(f'{out_folder}/png/post_processed/masks') and png:
        os.makedirs(f'{out_folder}/png/post_processed/masks')

    if not os.path.exists(f'{out_folder}/png/post_processed/concat') and png:
        os.makedirs(f'{out_folder}/png/post_processed/concat')

    if not os.path.exists(f'{out_folder}/nifti'):
        os.makedirs(f'{out_folder}/nifti')

    if not os.path.exists(f'{out_folder}/nifti/raw'):
        os.makedirs(f'{out_folder}/nifti/raw')

    if not os.path.exists(f'{out_folder}/nifti/post_processed'):
        os.makedirs(f'{out_folder}/nifti/post_processed')

    if not os.path.exists(f'{out_folder}/nifti/post_processed/images'):
        os.makedirs(f'{out_folder}/nifti/post_processed/images')

    if not os.path.exists(f'{out_folder}/nifti/post_processed/masks'):
        os.makedirs(f'{out_folder}/nifti/post_processed/masks')

    if not os.path.exists(f'{out_folder}/nifti/post_processed/masks_cca'):
        os.makedirs(f'{out_folder}/nifti/post_processed/masks_cca')

    if not os.path.exists(f'{out_folder}/nifti/post_processed/masks_binary'):
        os.makedirs(f'{out_folder}/nifti/post_processed/masks_binary')

    if not os.path.exists(f'{out_folder}/nifti/post_processed/concat'):
        os.makedirs(f'{out_folder}/nifti/post_processed/concat')

    if not os.path.exists(f'{out_folder}/nifti/post_processed/concat_binary'):
        os.makedirs(f'{out_folder}/nifti/post_processed/concat_binary')

    if not os.path.exists(f'{out_folder}/nifti/post_processed/concat_cca'):
        os.makedirs(f'{out_folder}/nifti/post_processed/concat_cca')




def generate_images(experiment, n_png, n_nifti, png, im_size, rescale_intensity, out_folder, threshold,
                    post_process, latent_dim, sbs):
    model_path = f'saved_models/{experiment}/saved_model'

    create_folders(out_folder)

    generated_png = 0
    xx = 0
    for x in tqdm(range(691,n_nifti)):

        gen_image = generate_image(model_path=model_path, img_size=im_size, rescale_intensity=rescale_intensity, latent_dim=latent_dim)

        if post_process:

            if sbs:
                x, y, z = gen_image.shape

                # create empty arrays for the two parts
                image = np.zeros((x, y, z // 2))
                mask = np.zeros((x, y, z // 2))

                # loop over the slices and add them to the appropriate part
                for i in range(z):
                    if i % 2 == 0:
                        image[:, :, i // 2] = gen_image[:, :, i]
                    else:
                        mask[:, :, (i - 1) // 2] = gen_image[:, :, i]

            else:
                image = gen_image[:, :, :im_size // 2]
                mask = gen_image[:, :, im_size // 2:]



            binary_mask = np.where(mask >= 0, 1, 0)

            if len(np.unique(binary_mask)) == 1:
                # Empty mask, disregard
               continue

            xx += 1
            if threshold:
                low_threshold = -1024
                high_threshold = 600
                image = image * (high_threshold - low_threshold) + low_threshold
                # image = image.astype(np.int16)

            # cca_mask = connected_component_analysis(mask, threshold=0, size_threshold=200)
            # if np.unique(cca_mask)[0] == 0 and len(np.unique(cca_mask)) == 1:
            #     # Discount images that don't have any masks
            #     continue
            #
            # cca_mask = np.where(cca_mask == 0, -1, 1)

            concat = np.concatenate((image, mask), axis=2)
            # concat_binary = np.concatenate((image, binary_mask), axis=2)
            # concat_cca = np.concatenate((image, cca_mask), axis=2)

            # nifti_binary_concat = nib.Nifti1Image(concat_binary, affine=np.eye(4))
            # nifti_cca_concat = nib.Nifti1Image(concat_cca, affine=np.eye(4))
            # nifti_concat = nib.Nifti1Image(concat, affine=np.eye(4))
            nifti_image = nib.Nifti1Image(image, affine=np.eye(4))
            # nifti_mask = nib.Nifti1Image(mask, affine=np.eye(4))
            nifti_mask_binary = nib.Nifti1Image(binary_mask, affine=np.eye(4))
            # nifti_mask_cca = nib.Nifti1Image(cca_mask, affine=np.eye(4))

            nib.save(nifti_mask_binary, f'{out_folder}/nifti/post_processed/masks_binary/gen_{xx + 691}.nii.gz')
            # nib.save(nifti_mask_cca, f'{out_folder}/nifti/post_processed/masks_cca/gen_{x + 1}.nii.gz')
            nib.save(nifti_image, f'{out_folder}/nifti/post_processed/images/gen_{xx+ 691}.nii.gz')
            # nib.save(nifti_mask, f'{out_folder}/nifti/post_processed//masks/gen_{xx}.nii.gz')
            # nib.save(nifti_concat, f'{out_folder}/nifti/post_processed/concat/gen_{x + 1}.nii.gz')
            # nib.save(nifti_binary_concat, f'{out_folder}/nifti/post_processed/concat_binary/gen_{x + 1}.nii.gz')
            # nib.save(nifti_cca_concat, f'{out_folder}/nifti/post_processed/concat_cca/gen_{x + 1}.nii.gz')

        im = nib.Nifti1Image(gen_image, affine=np.eye(4))
        nib.save(im, f'{out_folder}/nifti/raw/image_{x + 1}.nii.gz')

        if png and generated_png < n_png:
            fig = display_image(gen_image, show=False, return_figure=True)
            fig.savefig(f'{out_folder}/png/raw/image_{x + 1}.png')
            plt.close()

            if post_process:
                fig = display_image(concat, show=False, return_figure=True)
                fig.savefig(f'{out_folder}/png/post_processed/concat/image_{x + 1}.png')
                plt.close()

                fig = display_image(image, show=False, return_figure=True)
                fig.savefig(f'{out_folder}/png/post_processed/images/image_{x + 1}.png')
                plt.close()

                fig = display_image(mask, show=False, return_figure=True)
                fig.savefig(f'{out_folder}/png/post_processed/masks/image_{x + 1}.png')
            plt.close()

            generated_png += 1


if __name__ == '__main__':
    if not os.path.exists('data/generated_images'):
        os.makedirs('data/generated_images')
    n_png = 15
    n_nifti = 1500
    png = False
    nifti = True
    latent_dim = 1024
    im_size = 256
    rescale_intensity = False
    threshold = False
    mode = 'rescaled'
    post_process = True
    binary_mask = True
    sbs = False
    cca = False

    saved_models = [
        '256_normalized15mm',
        # '5.6.5 normalized15mm top1',
        # '256_reaffirm',
        # '256_hpo_17_non_norm',
        # '256_hpo_17_non_norm2',
        # '5.6.5 normalized15mm top1',
        # '5.8.1 cropped_non_normalized',
    ]

    for index, experiment in enumerate(saved_models):
        print(f'[{index + 1}/{len(saved_models)}] Generating images for experiment: {experiment}')
        out_folder = create_out_folders(experiment)
        generate_images(experiment=experiment, n_png=n_png, n_nifti=n_nifti, png=png, im_size=im_size,
                        rescale_intensity=rescale_intensity, out_folder=out_folder, threshold=threshold,
                        post_process=post_process, latent_dim=latent_dim, sbs=sbs)
