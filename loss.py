import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.morphology import binary_erosion, binary_dilation

class BoundaryAwareHybridLoss(nn.Module):
    def __init__(self, bce_weight=0.3, dice_weight=0.3, focal_weight=0.4, boundary_weight=0.5, pos_weight=torch.tensor([2.0]), gamma=2.0):
        super(BoundaryAwareHybridLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.gamma = gamma

    def dice_loss(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice_score

    def focal_loss(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        ce_loss = F.binary_cross_entropy(pred_flat, target_flat, reduction='none')
        p_t = pred_flat * target_flat + (1 - pred_flat) * (1 - target_flat)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()

    def boundary_loss(self, pred_boundary, target_change, smooth=1e-6):
        pred_boundary = torch.sigmoid(pred_boundary)
        # Generate ground truth boundary from change mask
        target_change_np = target_change.cpu().numpy()
        batch_size = target_change_np.shape[0]
        target_boundary = np.zeros_like(target_change_np)
        for i in range(batch_size):
            change_mask = target_change_np[i, 0] > 0.5
            boundary = binary_dilation(change_mask) ^ binary_erosion(change_mask)
            target_boundary[i, 0] = boundary.astype(float)
        target_boundary = torch.tensor(target_boundary, device=target_change.device)
        # Compute BCE for boundary (simple approach; can use distance transform as in search result [4])
        return F.binary_cross_entropy(pred_boundary, target_boundary)

    def forward(self, pred_change, pred_boundary, target_change):
        bce = self.bce_weight * self.bce_loss(pred_change, target_change)
        dice = self.dice_weight * self.dice_loss(pred_change, target_change)
        focal = self.focal_weight * self.focal_loss(pred_change, target_change)
        boundary = self.boundary_weight * self.boundary_loss(pred_boundary, target_change)
        change_loss = bce + dice + focal
        total_loss = change_loss + boundary
        return total_loss
