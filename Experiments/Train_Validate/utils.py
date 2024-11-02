
import torch
import torch.nn as nn
from math import log10

# Calculate PSNR
def calculate_psnr(img1, img2):
    mse = nn.functional.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * log10(1.0 / torch.sqrt(mse))

# Custom Weighted Loss Function
class WeightedLoss(nn.Module):
    def __init__(self, base_loss=nn.MSELoss(), weight_factor=0.8):
        super(WeightedLoss, self).__init__()
        self.base_loss = base_loss
        self.weight_factor = weight_factor

    def forward(self, output, target, mask):
        # Apply weighted loss: higher weight for non-masked areas
        loss = self.base_loss(output * (1 - mask), target * (1 - mask)) + \
               self.weight_factor * self.base_loss(output * mask, target * mask)
        return loss