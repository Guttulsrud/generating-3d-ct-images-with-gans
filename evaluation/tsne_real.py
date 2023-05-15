import glob
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from tqdm import tqdm
import matplotlib as mpl
from scipy.spatial import ConvexHull
mpl.use('TkAgg')
experiments = ['128_H17_binary_mask_non_interpolateBASELINE',
               '5.8.2 alt_concatenated_cropped_interpolated baseline',
               '5.6.5 normalized15mm top1'
               ]
real_images = glob.glob('../data/128/interpolated_resized/images/*.nii.gz')
real_samples = []
# real_images = real_images[:11]

groups = []
for real_img in tqdm(real_images):
    real_data = nib.load(real_img)
    real_samples.append(real_data.get_fdata())
    group = real_img.split('/')[-1].split('_')[0].split('images\\')[-1].replace('.nii.gz', '').split('-')[0]
    groups.append(group)

real_samples = np.array(real_samples)

# Perform t-SNE dimensionality reduction on the concatenated samples
tsne = TSNE(n_components=2, perplexity=10, learning_rate=200, random_state=42)
embeddings = tsne.fit_transform(real_samples.reshape(len(real_samples), -1))

groups = np.array(groups)
group_names = ['CHUM', 'CHUP', 'CHUS', 'CHUV', 'HGJ', 'HMR', 'MDA']

fig, ax = plt.subplots()

# plot each group individually
for group in group_names:
    ix = np.where(groups == group)
    print(f"Group: {group}, count: {len(ix[0])}")

    ax.scatter(embeddings[ix, 0], embeddings[ix, 1], label=group)

ax.grid(color='gray', linestyle='-', linewidth=0.3, alpha=0.8)

# Visualize the embeddings
plt.legend(loc='upper left')
plt.subplots_adjust(wspace=0.1, hspace=0.05, top=0.85, bottom=0.15, left=0, right=1)

plt.axis('off')
plt.savefig(f'tsne_real_only.png')
# plt.show()
