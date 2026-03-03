"""
=======================================================================
Base Dataset for Point Cloud Completion
=======================================================================

Template for creating new point cloud datasets.
All custom datasets should follow this interface to work with the unified
train.py and test.py scripts.

Usage:
    1. Copy this file as your starting point
    2. Implement __len__ and __getitem__
    3. Add your dataset to utils/data_loaders.py DATASET_LOADER_MAPPING

Example — custom dataset from paired .ply files:

    class MyCustomDataset(BaseCompletionDataset):
        def __init__(self, data_root, split='train'):
            super().__init__(data_root, split)
            self.samples = self._load_file_list()

        def _load_file_list(self):
            # Return list of (partial_path, gt_path, taxonomy_id, model_id) tuples
            ...

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            partial_path, gt_path, tax_id, model_id = self.samples[idx]
            partial = self.load_point_cloud(partial_path)
            gt = self.load_point_cloud(gt_path)
            return tax_id, model_id, {
                'partial_cloud': torch.from_numpy(partial).float(),
                'gtcloud': torch.from_numpy(gt).float()
            }
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from abc import abstractmethod


class BaseCompletionDataset(Dataset):
    """
    Abstract base dataset for point cloud completion.

    Return format:
        Each __getitem__ must return a tuple of:
            (taxonomy_id, model_id, data_dict)

        Where data_dict is a dictionary containing at minimum:
            - 'partial_cloud': torch.Tensor of shape (N, 3)
            - 'gtcloud':       torch.Tensor of shape (M, 3)

    This format is compatible with the existing collate_fn in
    utils/data_loaders.py and the unified train/test scripts.
    """

    def __init__(self, data_root=None, split='train', n_points=2048):
        """
        Args:
            data_root (str): Root directory of the dataset.
            split (str): One of 'train', 'val', 'test'.
            n_points (int): Number of input points to sample.
        """
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.n_points = n_points

    @abstractmethod
    def __len__(self):
        """Return the total number of samples."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        """
        Return a single sample.

        Returns:
            tuple: (taxonomy_id, model_id, data_dict)
                - taxonomy_id (str): Category identifier
                - model_id (str): Sample identifier
                - data_dict (dict): Must contain keys:
                    'partial_cloud': torch.Tensor (N, 3)
                    'gtcloud': torch.Tensor (M, 3)
        """
        raise NotImplementedError

    # ===================== Utility methods =====================

    @staticmethod
    def load_point_cloud(file_path):
        """
        Load a point cloud from various file formats.

        Supported: .pcd, .ply, .npy, .npz, .pts, .xyz, .txt

        Args:
            file_path (str): Path to point cloud file.

        Returns:
            np.ndarray: Point cloud of shape (N, 3).
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.npy':
            return np.load(file_path).astype(np.float32)

        elif ext == '.npz':
            data = np.load(file_path)
            # Try common key names
            for key in ['points', 'point_cloud', 'pc', 'arr_0']:
                if key in data:
                    return data[key].astype(np.float32)
            raise KeyError(f"No point cloud key found in {file_path}")

        elif ext in ('.pts', '.xyz', '.txt'):
            return np.loadtxt(file_path, dtype=np.float32)[:, :3]

        elif ext == '.ply':
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(file_path)
                return np.asarray(pcd.points, dtype=np.float32)
            except ImportError:
                raise ImportError("open3d is required to load .ply files")

        elif ext == '.pcd':
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(file_path)
                return np.asarray(pcd.points, dtype=np.float32)
            except ImportError:
                raise ImportError("open3d is required to load .pcd files")

        else:
            raise ValueError(f"Unsupported point cloud format: {ext}")

    @staticmethod
    def random_sample(pc, n_points):
        """
        Randomly sample n_points from a point cloud.

        Args:
            pc (np.ndarray): Point cloud (N, 3).
            n_points (int): Number of points to sample.

        Returns:
            np.ndarray: Sampled point cloud (n_points, 3).
        """
        n = pc.shape[0]
        if n >= n_points:
            idx = np.random.choice(n, n_points, replace=False)
        else:
            idx = np.random.choice(n, n_points, replace=True)
        return pc[idx]

    @staticmethod
    def normalize_point_cloud(pc):
        """
        Center and scale a point cloud to fit in a unit sphere.

        Args:
            pc (np.ndarray): Point cloud (N, 3).

        Returns:
            np.ndarray: Normalized point cloud (N, 3).
        """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc


# =====================================================================
# Example: Paired File Dataset Implementation
# =====================================================================

class PairedFileDataset(BaseCompletionDataset):
    """
    Example dataset that loads paired partial/complete point clouds
    from a directory structure:

        data_root/
            train/
                category_A/
                    model_001_partial.ply
                    model_001_complete.ply
                    ...
            test/
                ...

    This is provided as a reference implementation. Modify it for your
    own data structure.
    """

    def __init__(self, data_root, split='train', n_points=2048,
                 partial_suffix='_partial', complete_suffix='_complete',
                 file_ext='.ply'):
        super().__init__(data_root, split, n_points)
        self.partial_suffix = partial_suffix
        self.complete_suffix = complete_suffix
        self.file_ext = file_ext
        self.samples = self._build_file_list()

    def _build_file_list(self):
        """Build list of (partial_path, gt_path, taxonomy_id, model_id)."""
        samples = []
        split_dir = os.path.join(self.data_root, self.split)
        if not os.path.exists(split_dir):
            return samples

        for category in sorted(os.listdir(split_dir)):
            cat_dir = os.path.join(split_dir, category)
            if not os.path.isdir(cat_dir):
                continue

            # Find all partial files
            for f in sorted(os.listdir(cat_dir)):
                if f.endswith(self.partial_suffix + self.file_ext):
                    model_id = f.replace(self.partial_suffix + self.file_ext, '')
                    partial_path = os.path.join(cat_dir, f)
                    gt_path = os.path.join(
                        cat_dir, model_id + self.complete_suffix + self.file_ext
                    )
                    if os.path.exists(gt_path):
                        samples.append((partial_path, gt_path, category, model_id))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        partial_path, gt_path, taxonomy_id, model_id = self.samples[idx]

        partial = self.load_point_cloud(partial_path)
        gt = self.load_point_cloud(gt_path)

        # Sample to fixed number of points
        partial = self.random_sample(partial, self.n_points)

        return taxonomy_id, model_id, {
            'partial_cloud': torch.from_numpy(partial).float(),
            'gtcloud': torch.from_numpy(gt).float()
        }
