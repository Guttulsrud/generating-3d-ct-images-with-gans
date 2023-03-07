def get_model(config):
    img_size = config['img_size']
    latent_dim = config['latent_dim']
    num_class = config['num_class']

    if img_size == 256:
        from model.architecture.HA_GAN_256 import Discriminator, Generator, Encoder, Sub_Encoder
    else:
        from model.architecture.HA_GAN_128 import Discriminator, Generator, Encoder, Sub_Encoder

    G = Generator(mode='train', latent_dim=latent_dim, num_class=num_class).cuda()
    D = Discriminator(num_class=num_class).cuda()
    E = Encoder().cuda()
    Sub_E = Sub_Encoder(latent_dim=latent_dim).cuda()

    return G, D, E, Sub_E
