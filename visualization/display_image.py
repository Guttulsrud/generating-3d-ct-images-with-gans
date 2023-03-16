import math

import matplotlib as mpl

mpl.use('TkAgg')
from matplotlib import pyplot as plt


def display_image(data, colormap='gray', n_slices=False, show=True, return_figure=False):
    if not n_slices:
        n_slices = data.shape[2]

    n_cols = int(math.ceil(math.sqrt(n_slices)))
    n_rows = int(math.ceil(n_slices / float(n_cols)))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 8))

    for i, ax in enumerate(axes.flat):
        if i < n_slices:
            ax.imshow(data[:, :, i], cmap=colormap)
            ax.axis('off')
        else:
            ax.set_visible(False)

    plt.tight_layout()
    if show:
        plt.show()
    if return_figure:
        return fig
