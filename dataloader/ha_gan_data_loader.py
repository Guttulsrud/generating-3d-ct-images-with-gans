import torch
from dataloader.volume_dataset import Volume_Dataset


def inf_train_gen(data_loader):
    while True:
        for _, batch in enumerate(data_loader):
            yield batch


def get_data_loader(config):
    fold = config['fold']
    num_class = config['num_class']
    batch_size = config['batch_size']
    workers = config['workers']

    train_set = Volume_Dataset(data_dir='data/processed/training/', fold=fold, num_class=num_class)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               drop_last=True,
                                               shuffle=False,
                                               num_workers=workers)
    return inf_train_gen(train_loader)
