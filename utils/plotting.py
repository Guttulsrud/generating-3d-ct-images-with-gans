import os

from matplotlib import pyplot as plt


def save_images(fake_image, real_image, epoch, path, config):
    os.mkdir(f'{path}/epochs/{epoch}')
    generated_plot = create_plot(fake_image, config=config, title=f'Epoch {epoch} - Generated image')
    real_plot = create_plot(real_image, config=config, title=f'Epoch {epoch} - Real image')

    generated_plot.savefig(f'{path}/epochs/{epoch}/generated.png')
    real_plot.savefig(f'{path}/epochs/{epoch}/real.png')
    plt.close(generated_plot)
    plt.close(real_plot)


def create_plot(input_image, title, config):

    size = tuple(config['images']['shape'])
    image = None
    mask = None
    fig, axs = plt.subplots(9, 6, figsize=(15, 20))

    if size == (154, 154, 54):
        image = input_image[:, :, :27]
        mask = input_image[:, :, 27:54]
        z_range = 27
    if size == (78, 78, 78):
        image = input_image[:, :, :39]
        mask = input_image[:, :, 39:78]
        z_range = 39
    if size == (38, 38, 38):
        image = input_image[:, :, :19]
        mask = input_image[:, :, 19:38]
        z_range = 19

    for i in range(z_range):
        row = i // 3
        col = (i % 3) * 2

        axs[row, col].imshow(image[:, :, i], cmap='gray')
        axs[row, col].set_title(f"Slice {i + 1}", size=15)
        axs[row, col].axis('off')

        axs[row, col + 1].imshow(mask[:, :, i], cmap='gray')
        axs[row, col + 1].set_title(f"Slice {i + 1}", size=15)
        axs[row, col + 1].axis('off')

    fig.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig
