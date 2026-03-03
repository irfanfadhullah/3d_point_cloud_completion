"""
Dummy Dataset for Pipeline Verification.

Generates random point clouds for testing that the entire train/test
pipeline works without needing real data files.

Usage:
    # Standalone
    from dataset.DummyDataset import DummyDataset
    dataset = DummyDataset(n_samples=100, n_input=2048, n_gt=16384)

    # Via train.py --dummy
    python train.py --config configs/pcn.yaml --dummy
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    """
    Generates random partial and complete point clouds for pipeline testing.

    Each sample returns:
        (taxonomy_id, model_id, {'partial_cloud': (N_in, 3), 'gtcloud': (N_gt, 3)})

    The point clouds are random but normalized — partial is a subset of
    the complete cloud, making it a realistic proxy.
    """

    def __init__(self, n_samples=100, n_input=2048, n_gt=16384, n_categories=8):
        """
        Args:
            n_samples:    Number of samples in the dataset.
            n_input:      Number of points in the partial cloud.
            n_gt:         Number of points in the ground truth cloud.
            n_categories: Number of dummy categories.
        """
        super().__init__()
        self.n_samples = n_samples
        self.n_input = n_input
        self.n_gt = n_gt
        self.n_categories = n_categories

        # Pre-generate categories
        self.categories = [f'cat_{i:04d}' for i in range(n_categories)]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Generate a random "complete" point cloud (unit sphere)
        gt = np.random.randn(self.n_gt, 3).astype(np.float32)
        gt = gt / np.max(np.sqrt(np.sum(gt ** 2, axis=1, keepdims=True)))

        # Partial = random subset of gt
        subset_idx = np.random.choice(self.n_gt, self.n_input, replace=False)
        partial = gt[subset_idx].copy()

        taxonomy_id = self.categories[idx % self.n_categories]
        model_id = f'model_{idx:06d}'

        return taxonomy_id, model_id, {
            'partial_cloud': torch.from_numpy(partial).float(),
            'gtcloud': torch.from_numpy(gt).float(),
        }


class DummyMVPDataset(Dataset):
    """
    Dummy MVP-format dataset for pipeline testing.

    Returns (label, partial, gt) matching the MVP data loader format.
    """

    def __init__(self, n_samples=100, n_input=2048, n_gt=2048):
        super().__init__()
        self.n_samples = n_samples
        self.n_input = n_input
        self.n_gt = n_gt

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        gt = np.random.randn(self.n_gt, 3).astype(np.float32)
        gt = gt / np.max(np.sqrt(np.sum(gt ** 2, axis=1, keepdims=True)))

        subset_idx = np.random.choice(self.n_gt, self.n_input, replace=False)
        partial = gt[subset_idx].copy()

        label = idx % 16

        return label, torch.from_numpy(partial).float(), torch.from_numpy(gt).float()


class DummyShapeNet55Dataset(Dataset):
    """
    Dummy ShapeNet55-format dataset for pipeline testing.

    Returns (taxonomy_id, model_id, {'gtcloud': (N, 3)}) —
    partial clouds are generated online in the trainer.
    """

    def __init__(self, n_samples=100, n_gt=8192):
        super().__init__()
        self.n_samples = n_samples
        self.n_gt = n_gt

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        gt = np.random.randn(self.n_gt, 3).astype(np.float32)
        gt = gt / np.max(np.sqrt(np.sum(gt ** 2, axis=1, keepdims=True)))

        taxonomy_id = f'cat_{idx % 55:04d}'
        model_id = f'model_{idx:06d}'

        return taxonomy_id, model_id, {
            'gtcloud': torch.from_numpy(gt).float(),
        }
