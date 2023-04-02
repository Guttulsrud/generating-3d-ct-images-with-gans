import glob

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from skimage.transform import resize
from tqdm import tqdm

real_images = glob.glob('../data/original/images/*.nii.gz')
fake_images = glob.glob('../data/original/images/*.nii.gz')

real_images = real_images[:10]
fake_images = fake_images[:10]

real_samples = []
for real_img in tqdm(real_images):
    real_data = nib.load(real_img)
    data = resize(real_data.get_fdata(), (128, 128, 128), mode='constant', cval=-1)
    real_samples.append(data)

real_samples = np.array(real_samples)

fake_samples = []
for fake_img in tqdm(fake_images):
    fake_data = nib.load(fake_img)
    data = resize(fake_data.get_fdata(), (128, 128, 128), mode='constant', cval=-1)

    fake_samples.append(data)
fake_samples = np.array(fake_samples)

# Concatenate real and fake samples
all_samples = np.concatenate((real_samples, fake_samples))

# Perform t-SNE dimensionality reduction on the concatenated samples
tsne = TSNE(n_components=2, perplexity=10, learning_rate=200)
embeddings = tsne.fit_transform(all_samples.reshape(len(all_samples), -1))

# Visualize the embeddings
plt.scatter(embeddings[:len(real_samples), 0], embeddings[:len(real_samples), 1], label='Real')
plt.scatter(embeddings[len(real_samples):, 0], embeddings[len(real_samples):, 1], label='Fake')
plt.legend()
plt.show()
