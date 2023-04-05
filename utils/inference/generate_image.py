import torch
from utils.get_model import get_model
from utils.ha_gan_utils import trim_state_dict_name


def generate_image(model_path, save_step=80000, rescale_intensity=True, img_size=128):
    batch_size = 1
    latent_dim = 1024
    generator, _, encoder, sub_encoder, = get_model(img_size=img_size,
                                                    latent_dim=latent_dim,
                                                    num_class=0,
                                                    mode='eval')

    ckpt_path = f"{model_path}/G_iter{str(save_step)}.pth"
    ckpt = torch.load(ckpt_path)['model']
    ckpt = trim_state_dict_name(ckpt)
    generator.load_state_dict(ckpt)

    ckpt_path = f"{model_path}/E_iter{str(save_step)}.pth"
    ckpt = torch.load(ckpt_path)['model']
    ckpt = trim_state_dict_name(ckpt)
    encoder.load_state_dict(ckpt)

    ckpt_path = f"{model_path}/Sub_E_iter{str(save_step)}.pth"
    ckpt = torch.load(ckpt_path)['model']
    ckpt = trim_state_dict_name(ckpt)
    sub_encoder.load_state_dict(ckpt)
    del ckpt

    generator = generator.cuda()
    encoder = encoder.cuda()
    sub_encoder = sub_encoder.cuda()

    generator.eval()
    encoder.eval()
    sub_encoder.eval()

    torch.cuda.empty_cache()

    with torch.no_grad():
        z_rand = torch.randn((batch_size, latent_dim)).cuda()
        x_rand = generator(z_rand, 0)
        x_rand = x_rand.detach().cpu().numpy()
        if rescale_intensity:
            x_rand = 0.5 * x_rand + 0.5  # rescale intensity to [0,1]

        x_rand = x_rand[0, 0, :, :, :]
        return x_rand
