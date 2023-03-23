
def get_model(img_size, latent_dim, num_class, mode='train'):
    if img_size == 256:
        from model.Model_HA_GAN_256 import Discriminator, Generator, Encoder, Sub_Encoder
    else:
        from model.Model_HA_GAN_128 import Discriminator, Generator, Encoder, Sub_Encoder

    G = Generator(mode=mode, latent_dim=latent_dim, num_class=num_class).cuda()
    D = Discriminator(num_class=num_class).cuda()
    E = Encoder().cuda()
    Sub_E = Sub_Encoder(latent_dim=latent_dim).cuda()
    return G, D, E, Sub_E
