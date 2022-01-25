import torch
from nets.vit import ViT
from nets.mae import MAE
from vis_tools import Visualizer
import numpy as np
import random
from tqdm import tqdm
from dataloader import DatasetTrain
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


def to01(img):
    return np.clip(img, a_min=0, a_max=1)


def train(start_epoch):
    train_vis = Visualizer(env='training_mae')
    dataset = DatasetTrain()
    v = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048,
        channels=1
    ).cuda()

    mae = MAE(
        encoder=v,
        masking_ratio=0.75,
        decoder_dim=512,
        decoder_depth=6
    ).cuda()

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True
    )
    optimizer = torch.optim.Adam(mae.parameters(), lr=1e-5, betas=(0.5, 0.999))

    for epoch in range(start_epoch + 1, start_epoch + 101):
        epoch_loss = 0

        for i, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            img = batch.cuda()
            loss, recon = mae(img)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            train_vis.img('img', img.detach().cpu())
            train_vis.img('recon', to01(recon.detach().cpu()))

        train_vis.plot('loss', epoch_loss)
        print('epoch：[%d] loss: [%.4f]' % (epoch, epoch_loss))


if __name__ == '__main__':
    train(0)
