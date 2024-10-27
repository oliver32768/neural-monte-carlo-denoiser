import torch
import torch.nn.functional as F
import numpy as np

class LoG():
    def __init__(self, sigma, device):
        self.device = device
        self.init_kernel(sigma=sigma, device=device)

    def init_kernel(self, sigma, device):
        C_in = 3
        C_out = 3

        x, y = np.meshgrid(np.linspace(-4, 4, 9), np.linspace(-4, 4, 9))
        self.LoG_kernel = np.array(1.0 / (np.pi * (sigma**4)) * ((((x**2) + (y**2)) / (2 * (sigma**2))) - 1) * np.e**(-((x**2) + (y**2)) / (2 * (sigma**2))))
        self.LoG_kernel *= sigma**2

        self.LoG_kernel = torch.tensor(self.LoG_kernel, dtype=torch.float32, requires_grad=False).repeat(C_out, C_in, 1, 1).to(device)

    def __call__(self, img):
        # Passing 5D now - img = N x M x C x H x W
        N, M, C, H, W = img.shape
        LoG_img = torch.zeros(N, M, C, H, W, device=self.device)
        for j in range(M):
            LoG_img[:, j] = F.conv2d(input=img[:, j], weight=self.LoG_kernel, padding='same')
        return LoG_img