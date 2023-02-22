import os

from matplotlib import pyplot as plt


def save_images(fake_image, real_image, epoch, path):
    os.mkdir(f'{path}/epochs/{epoch}')
    generated_plot = create_plot(fake_image, title=f'Epoch {epoch} - Generated image')
    real_plot = create_plot(real_image, title=f'Epoch {epoch} - Real image')

    generated_plot.savefig(f'{path}/epochs/{epoch}/generated.png')
    real_plot.savefig(f'{path}/epochs/{epoch}/real.png')


def separate_mask(input_image):
    image = input_image[:, :, :27]
    mask = input_image[:, :, 27:54]
    return image, mask


def create_plot(image, title):
    fig, axs = plt.subplots(9, 6, figsize=(15, 20))
    image, mask = separate_mask(image)

    for i in range(27):
        row = i // 3
        col = (i % 3) * 2

        axs[row, col].imshow(image[:, :, i], cmap='viridis')
        axs[row, col].set_title(f"Slice {i + 1}", size=15)
        axs[row, col].axis('off')

        axs[row, col + 1].imshow(mask[:, :, i], cmap='viridis')
        axs[row, col + 1].set_title(f"Slice {i + 1}", size=15)
        axs[row, col + 1].axis('off')

    fig.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig
