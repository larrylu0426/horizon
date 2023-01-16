import cv2
import numpy as np
from PIL import Image
import torch
from torch import Tensor
import torchvision


def get_masked_image(image: Tensor, mean, std, attn: Tensor):
    _, H, W = image.shape

    t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(3, H, W)
    t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, H, W)
    image = image * t_std + t_mean
    image = torchvision.transforms.functional.to_pil_image(image)
    image = np.asarray(image)

    heatmap = attn.numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (W, H))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    heatmap = np.asarray(heatmap)
    masked_image = heatmap * 0.4 + image

    return masked_image