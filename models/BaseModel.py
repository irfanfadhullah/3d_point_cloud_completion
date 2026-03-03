"""
=======================================================================
Base Model for Point Cloud Completion
=======================================================================

Template for creating new point cloud completion models.
All models should follow this interface to work with the unified
train.py and test.py scripts.

Usage:
    1. Copy this file as your starting point
    2. Implement __init__, forward, build_loss_func, and get_loss
    3. Register your model with @MODELS.register_module()

Example:
    from models.BaseModel import BaseModel
    from models.build import MODELS

    @MODELS.register_module()
    class MyNewModel(BaseModel):
        def __init__(self, config):
            super().__init__(config)
            self.encoder = ...
            self.decoder = ...
            self.build_loss_func()

        def forward(self, xyz):
            # xyz: (B, N, 3) partial point cloud
            coarse = ...
            fine = ...
            return (coarse, fine)

        def build_loss_func(self):
            self.loss_func = ChamferDistanceL2()

        def get_loss(self, ret, gt, epoch=0):
            loss_coarse = self.loss_func(ret[0], gt)
            loss_fine = self.loss_func(ret[1], gt)
            return loss_coarse + loss_fine
"""

import torch
import torch.nn as nn
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Abstract base class for point cloud completion models.

    Contract:
        - __init__(config): Initialize the model with a config object (EasyDict or similar).
        - forward(xyz) -> tuple: Takes partial point cloud (B, N, 3) and returns
          a tuple of predicted point clouds, ordered from coarse to fine.
          Each element is (B, M_i, 3).
        - build_loss_func(): Set up loss functions (e.g., ChamferDistance).
        - get_loss(ret, gt, epoch) -> loss or tuple of losses: Compute loss
          between predictions and ground truth.

    The unified train.py will call:
        pcds_pred = model(partial)         # forward pass
        loss = model.get_loss(pcds_pred, gt, epoch)  # if model has get_loss

    If the model doesn't implement get_loss, the framework falls back to the
    default pyramid chamfer distance loss.
    """

    def __init__(self, config=None):
        super(BaseModel, self).__init__()
        self.config = config

    @abstractmethod
    def forward(self, xyz):
        """
        Forward pass for point cloud completion.

        Args:
            xyz (torch.Tensor): Partial point cloud of shape (B, N, 3).

        Returns:
            tuple: Predicted point clouds ordered coarse to fine.
                   Each element is a Tensor of shape (B, M_i, 3).
                   Minimum 2 elements: (coarse, fine).

        Example return:
            return (coarse, fine)
            return (P0, P1, P2, P3)
        """
        raise NotImplementedError

    def build_loss_func(self):
        """
        Initialize loss function(s) used in get_loss().
        Override this to set up custom loss functions.

        Example:
            def build_loss_func(self):
                from extensions.chamfer_dist import ChamferDistanceL2
                self.loss_func = ChamferDistanceL2()
        """
        pass

    def get_loss(self, ret, gt, epoch=0):
        """
        Compute loss between predictions and ground truth.

        Args:
            ret (tuple): Output of forward(), tuple of predicted point clouds.
            gt (torch.Tensor): Ground truth complete point cloud (B, M, 3).
            epoch (int): Current epoch number (useful for loss scheduling).

        Returns:
            torch.Tensor or tuple: Total loss, or (total_loss, loss_dict).

        If not overridden, the unified trainer will use the default
        pyramid chamfer distance loss.
        """
        raise NotImplementedError(
            "Model does not implement get_loss(). "
            "The unified trainer will use the default pyramid CD loss instead."
        )
