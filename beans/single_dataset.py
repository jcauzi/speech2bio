from torch.utils.data import Dataset
import os
import torch
import random

class LayerNonBatchDataset(Dataset):
    def __init__(self, data_dir, labels_path, layer_num):
        """
        data_dir : Path to the dataset directory ("DATASET/train/")
        labels_path : Path to the labels file ("DATASET/train/labels.pt")
        layer_num : The layer index to load (0-based index).
        """

        self.data_dir = data_dir
        self.layer_folder = os.path.join(data_dir, f"layer_{layer_num}")
        self.labels = torch.load(labels_path)  # Load all labels into memory

        if not os.path.exists(self.layer_folder):
            raise FileNotFoundError(f"Layer folder not found: {self.layer_folder}")

        # Collect all example files
        self.sample_files = sorted(
            [os.path.join(self.layer_folder, f) for f in os.listdir(self.layer_folder) if f.startswith("sample_")],
            key=lambda x: int(x.split("_")[-1].split(".")[0])  # Sort by index
        )

        # Ensure total size of labels = number of saved samples
        if len(self.sample_files) != len(self.labels):
            raise ValueError(f"Total labels ({len(self.labels)}) != number of saved samples ({len(self.sample_files)})")

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.sample_files):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.sample_files)} samples")

        sample_path = self.sample_files[idx]
        sample_data = torch.load(sample_path)  # Load individual sample

        return sample_data, self.labels[idx]
