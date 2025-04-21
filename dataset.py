from torch.utils.data import Dataset
import torch
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms as pth_transforms
from pathlib import Path
import utils


class TacForceDataset(Dataset):
    """Dataset class for tactile-force estimation and statistics computation."""

    def __init__(self, args, split, transform=None, transform_depth=None,  mode="standard"):
        """
        Arguments:
            args: Argument parser with dataset paths and settings.
            split (string): Dataset split ("train", "val", "test").
            transform (callable, optional): Transform for tactile images.
            transform_depth (callable, optional): Transform for depth images.
            mode (string): "standard" for normal operation, "compute_stats" for mean/std computation.
        """
        labels_path = Path(args.data_basedir) / vars(args)[f"labels_{split}"]
        self.df = pd.read_csv(labels_path)
        self.df.dropna(inplace=True)
        self.df["depth"] = args.data_basedir + self.df["depth"]
        self.df["tactile"] = args.data_basedir + self.df["tactile"]

        self.transform = transform
        self.transform_depth = transform_depth
        # Precomputed min/max for depth images
        self.max, self.min = 27.03226967220051, -8.311515796355053
        self.bias = args.bias
        self.tactile_mode = args.tactile_mode
        self.input_modality = args.input_modality
        
        if mode not in ["standard", "compute_stats"]:
            raise ValueError("Invalid mode. Choose 'standard' or 'compute_stats'.")
        
        self.mode = mode  # "standard" or "compute_stats"

        # Only compute min/max for depth normalization in compute_stats mode.
        # Then you need to change the self.max and self.min values to the ones you computed
        # or using utils.find_max_and_min(self.df["depth"])
        if self.mode == "compute_stats":
            self.max, self.min = utils.find_max_and_min(self.df["depth"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load tactile image
        tactile_path = self.df["tactile"].iloc[idx]
        if self.tactile_mode == "nobg":
            tactile_path = tactile_path.replace("tactile", "tactile_nobg")
        
        tactile = Image.open(tactile_path)

        # Load and normalize depth image
        depth = np.load(self.df["depth"].iloc[idx])
        depth = (depth - (self.min - self.bias)) / (2 * self.bias + self.max - self.min)
        depth = Image.fromarray((np.repeat(depth[:, :, np.newaxis], 3, axis=2) * 255).astype('uint8'))

        # Mode for mean and std computation (return only images)
        if self.mode == "compute_stats":
            tactile = pth_transforms.Pad([0, 40, 0, 40])(tactile)
            tactile = pth_transforms.Resize(224)(tactile)
            tactile = pth_transforms.ToTensor()(tactile)

            depth = pth_transforms.Pad([0, 40, 0, 40])(depth)
            depth = pth_transforms.Resize(224)(depth)
            depth = pth_transforms.ToTensor()(depth)

            return tactile, depth  # Only return images for mean/std computation
        
        # Load force-torque data
        ft = self.df["FT"].iloc[idx]
        ft = np.fromstring(ft, dtype=float, sep=' ')
        ft = torch.from_numpy(ft).type(torch.FloatTensor)[:3]  # Force vector only
        
        # Apply transforms if needed
        if self.transform:
            tactile = self.transform(tactile)
        if self.transform_depth:
            depth = self.transform_depth(depth)
        
        return tactile, ft, depth