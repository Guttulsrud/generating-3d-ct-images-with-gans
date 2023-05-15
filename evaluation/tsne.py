import glob
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from tqdm import tqdm
import matplotlib as mpl

mpl.use('TkAgg')

experiments = ['128_H17_binary_mask_non_interpolateBASELINE',
               '5.8.2 alt_concatenated_cropped_interpolated baseline',
               '5.6.5 normalized15mm top1'
               ]
real_images = glob.glob('../data/128/interpolated_resized/images/*.nii.gz')

real_samples = []
for real_img in tqdm(real_images):
    real_data = nib.load(real_img)
    real_samples.append(real_data.get_fdata())
real_samples = np.array(real_samples)

for experiment in experiments:
    fake_images = glob.glob(f'../data/generated_images/{experiment}/nifti/post_processed/images/*.nii.gz')

    fake_samples = []
    for fake_img in tqdm(fake_images):
        fake_data = nib.load(fake_img)
        fake_samples.append(fake_data.get_fdata())

    fake_samples = np.array(fake_samples)

    # Perform t-SNE on real_samples
    tsne = TSNE(n_components=2, perplexity=10, learning_rate=200, random_state=42)
    real_embeddings = tsne.fit_transform(real_samples.reshape(len(real_samples), -1))

    # Plot real samples with a specific color
    plt.scatter(real_embeddings[:, 0], real_embeddings[:, 1], c=[(0.3, 0.6, 1.0)], label='Real')

    # Perform t-SNE on generated_samples
    generated_embeddings = tsne.fit_transform(fake_samples.reshape(len(fake_samples), -1))

    # Plot generated samples with a different color
    plt.scatter(generated_embeddings[:, 0], generated_embeddings[:, 1], c='orange', label='Generated')

    plt.legend(loc='upper left')
    plt.subplots_adjust(wspace=0.1, hspace=0.05, top=0.85, bottom=0.15, left=0, right=1)

    plt.axis('off')
    plt.savefig(f'tsne_{experiment}.png')
    plt.close()
    # plt.show()
    # exit()
