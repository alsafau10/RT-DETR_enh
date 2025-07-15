import torch
import numpy as np
import cv2


import kornia.enhance as KE
from typing import Tuple

class HistogramEqualization:
    """
    Apply per-channel histogram equalization on a batch of images.

    images: torch.Tensor[B, C, H, W], values in [0..1]
    returns: torch.Tensor[B, C, H, W]
    """
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        eq_imgs = []
        for img in images:
            arr = (img.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            he = np.stack([cv2.equalizeHist(arr[:, :, c]) for c in range(arr.shape[2])], axis=-1)
            t = torch.from_numpy(he.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
            eq_imgs.append(t)
        return torch.stack(eq_imgs, dim=0)


class LinearStretch:
    """
    Apply linear contrast stretching on a batch of images.

    images: torch.Tensor[B, C, H, W]
    returns: torch.Tensor[B, C, H, W]
    """
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        stretched = []
        for img in images:
            cmin = img.amin(dim=[1, 2], keepdim=True)
            cmax = img.amax(dim=[1, 2], keepdim=True)
            img_st = (img - cmin) / (cmax - cmin + self.eps)
            stretched.append(img_st)
        return torch.stack(stretched, dim=0)


class CLAHE:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) channel-wise.

    clip_limit: float
    tile_grid_size: tuple
    """
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        out = []
        for img in images:
            arr = (img.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            clahed = np.stack([self.clahe.apply(arr[:, :, c]) for c in range(arr.shape[2])], axis=-1)
            t = torch.from_numpy(clahed.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
            out.append(t)
        return torch.stack(out, dim=0)
        




class GPUHistogramEqualization:
    """
    GPU histogram equalization for a batch of images.

    images: torch.Tensor[B, C, H, W], values in [0..1]
    returns: torch.Tensor[B, C, H, W]
    """
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        # Scale images to [0..255]
        imgs255 = images * 255.0
        # GPU-native histogram equalization via Kornia
        eq = KE.equalize_hist(imgs255)
        # Convert back to [0..1]
        return eq / 255.0


class GPUCLAHE:
    """
    GPU CLAHE via Kornia's AdaptiveHistogramEqualization.

    clip_limit: float
    tile_grid_size: tuple
    """
    def __init__(self, clip_limit : int | float = 20, grid_size : Tuple[int,int] = (8,8) ) -> None:
        self.clip_limit, self.grid_size = float(clip_limit), grid_size

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        # Convert back to [0..1]
        return KE.equalize_clahe(images, self.clip_limit, self.grid_size)



class LinearStretch:
    """
    Channel-wise linear contrast stretching for [B, C, H, W].
    GPU-native since pure PyTorch.
    """
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        cmin = images.amin(dim=[2, 3], keepdim=True)
        cmax = images.amax(dim=[2, 3], keepdim=True)
        return (images - cmin) / (cmax - cmin + self.eps)

