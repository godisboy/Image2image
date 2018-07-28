import torch
import torch.nn as nn


class MMDLoss(nn.Module):
    def __init__(self):
        super(MMDLoss, self).__init__()

    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        tiled_x = x.view(x_size, 1, dim).repeat(1, x_size, 1)
        tiled_y = y.view(1, y_size, dim).repeat(y_size, 1, 1)
        out = torch.exp(-(1)*torch.mean(torch.pow((tiled_x - tiled_y), 2), dim=2) / dim)
        # print(out.size())
        return out

    def compute_mmd(self, x, y):
        if len(x.size()) > 2:
            x = x.view(x.size(0), -1)
            y = y.view(y.size(0), -1)

        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)

    def forward(self, latent_a, latent_b):
        loss_mmd = self.compute_mmd(latent_a, latent_b)
        return loss_mmd

# mmd = MMD()
# latent_a = torch.randn((1, 100, 64*64))
# latent_b = torch.randn((1, 100, 64*64))
# loss_mmd = mmd(latent_a, latent_b)
# print(loss_mmd)