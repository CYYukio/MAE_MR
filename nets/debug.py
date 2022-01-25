import torch

from nets.vit import ViT
from nets.mae import MAE
from torchsummary import summary

if __name__ == '__main__':
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
        masking_ratio=0.75,   # the paper recommended 75% masked patches
        decoder_dim=512,      # paper showed good results with just 512
        decoder_depth=6       # anywhere from 1 to 8
    ).cuda()
    # summary(mae, input_size=(1,256,256))
    img = torch.zeros((1,1,256,256)).cuda()
    out = mae(img)

