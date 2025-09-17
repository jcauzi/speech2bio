import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class LayerBatchDataset(Dataset):
    def __init__(self, data_dir, labels_path, layer_num):
        """
            data_dir : path to the dataset directory ("DATASET/train/")
            labels_path : Path to the labels file ("DATASET/train/labels.pt")
            layer_num : The layer index to load (0 based index).
        """

        self.data_dir = data_dir
        self.layer_folder = os.path.join(data_dir, f"layer_{layer_num}")
        self.labels = torch.load(labels_path)  # this loads all labels into memory
        print(f"total labels loaded: {len(self.labels)}")

        if not os.path.exists(self.layer_folder):
            raise FileNotFoundError(f"Layer folder not found: {self.layer_folder}")

        #collect batch files for the layer :
        self.batch_files = sorted(
            [os.path.join(self.layer_folder, f) for f in os.listdir(self.layer_folder) if f.startswith("batch_")],
            key=lambda x: int(x.split("_")[-1].split(".")[0])  #sorted by batch index
        )

        # compute batch sizes to slice labels dynamically :
        print("Extracting batch sizes...")
        self.batch_sizes = []
        for batch_file in tqdm(self.batch_files, desc="Processing batches"):
            batch_data = torch.load(batch_file)
            
            # Ensure consistency in batch shape
            if batch_data.dim() == 4 and batch_data.shape[0] == 1:
                batch_data = batch_data.squeeze(0)  #remove leading dimension
            
            assert batch_data.dim() == 3, f"invalid batch shape: {batch_data.shape}, expected [batch_size, time_steps, features]"
            
            self.batch_sizes.append(batch_data.shape[0])  # First dimension is batch size

        # Ensure total size of labels = sum of batch sizes
        if sum(self.batch_sizes) != len(self.labels):
            raise ValueError(
                f"total labels ({len(self.labels)}) != sum of batch sizes ({sum(self.batch_sizes)})"
            )

    def __len__(self):
        # Number of batch files 
        return len(self.batch_files)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.batch_files):
            raise IndexError(f"index {idx} out of range for dataset with {len(self.batch_files)} batches")

        batch_path = self.batch_files[idx]
        batch_data = torch.load(batch_path)
        # slice labels :
        start_idx = sum(self.batch_sizes[:idx])
        end_idx = start_idx + self.batch_sizes[idx]
        batch_labels = self.labels[start_idx:end_idx]  # shape: [batch_size, ...]

        return batch_data, batch_labels
