import torch

_ = torch.manual_seed(123)
from torchmetrics.image.kid import KernelInceptionDistance

kid = KernelInceptionDistance()
# generate two slightly overlapping image intensity distributions
imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
kid.update(imgs_dist1, real=True)
kid.update(imgs_dist2, real=False)
kid_mean, kid_std = kid.compute()
print((kid_mean, kid_std))
