import numpy as np
from matplotlib import pyplot as plt


def plot_center_sagittal_slice(name, data, return_fig=False, show=True):
    data = np.transpose(data, axes=(1, 0, 2))
    center_sagittal_index = data.shape[1] // 2
    center_sagittal_slice = data[:, center_sagittal_index, :]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(np.rot90(center_sagittal_slice), cmap='gray', aspect='equal',
                   extent=[0, data.shape[0], 0, data.shape[0]])
    ax.set_title(f'{name}')
    ax.axis('off')

    if show:
        plt.show()
    if return_fig:
        return fig