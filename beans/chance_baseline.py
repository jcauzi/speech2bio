# from collections import Counter
# import numpy as np
# import torch

# # def compute_class_proportions_from_dataloader(dataloader):
# #     """
# #     Compute class proportions based on labels in the dataloader.

# #     Args:
# #         dataloader (DataLoader): A dataloader that provides sliding windows and their labels.

# #     Returns:
# #         dict: Class proportions {label: proportion}.
# #     """
# #     label_counts = Counter()
# #     total_windows = 0

# #     for _, labels in dataloader:  # Assuming dataloader yields (data, labels)
# #         total_windows += len(labels)
# #         for label_set in labels:  # Each `label_set` contains labels for a window
# #             label_counts.update(label_set)

# #     # Calculate class proportions
# #     total_labels = sum(label_counts.values())
# #     class_proportions = {label: count / total_labels for label, count in label_counts.items()}
# #     return class_proportions

# # def compute_class_proportions_from_dataloader(dataloader):
# #     """
# #     Compute class proportions from the binary labels output by the dataloader.

# #     Args:
# #         dataloader (DataLoader): The dataloader object.

# #     Returns:
# #         dict: Class proportions {class_index: proportion}.
# #     """
# #     total_labels = torch.zeros(dataloader.dataset[0][1].shape[1])  # Initialize count for each class
# #     total_windows = 0

# #     for _, labels in dataloader:  # Iterates through batches
# #         total_labels += labels.sum(dim=0)  # Sum over batch dimension
# #         total_windows += labels.size(0)   # Count windows in the batch

# #     class_proportions = total_labels / total_windows
# #     return class_proportions

# def compute_class_proportions_from_dataloader(dataloader):
#     """
#     Compute class proportions from the binary labels output by the dataloader.

#     Args:
#         dataloader (DataLoader): The dataloader object.

#     Returns:
#         dict: Class proportions {class_index: proportion}.
#     """

#     first_sample = dataloader.dataset[0]
#     num_classes = first_sample[1].shape[0]

#     total_labels = torch.zeros(num_classes)
#     total_windows = 0

#     for _, labels in dataloader:  # Iterates through batches
#         total_labels += labels.sum(dim=0)  # Sum over batch dimension
#         total_windows += labels.size(0)   # Count windows in the batch

#     # Handle the absence class (all-zero rows)
#     absence_count = total_windows - total_labels.sum().item()
#     total_labels = torch.cat([total_labels, torch.tensor([absence_count])])

#     class_proportions = total_labels / total_windows
#     return class_proportions


# def compute_chance_map(class_proportions):
#     """
#     Compute chance MAP based on class proportions.

#     Args:
#         class_proportions (dict): Class proportions {label: proportion}.

#     Returns:
#         float: Chance MAP.
#     """
#     return (class_proportions / 2).mean().item()

# # Main function
# def chance_baseline(dataloader):
#     # Compute class proportions
#     class_proportions = compute_class_proportions_from_dataloader(dataloader)
    
#     # Compute chance MAP
#     chance_map = compute_chance_map(class_proportions)
#     print(f"Class Proportions: {class_proportions.numpy()}")
#     print(f"Chance MAP: {chance_map:.4f}")

#     # Compute most frequent MAP
#     map_most_frequent = np.max(class_proportions)
#     print(f"Most frequent MAP : {map_most_frequent}")
#     return class_proportions.numpy()

# # Usage
# # Assuming `dataloader` is your preprocessed dataloader
# # main(dataloader)













import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_class_proportions_from_dataloader(dataloader):
    """
    Compute class proportions from the binary labels output by the dataloader.
    Only consider the positive classes (exclude the absence of label class).

    Args:
        dataloader (DataLoader): The dataloader object.

    Returns:
        torch.Tensor: Class proportions for positive classes.
    """

    first_sample = dataloader.dataset[0]
    print(first_sample)
    num_classes = first_sample[1].shape[0]  # Number of classes

    total_labels = torch.zeros(num_classes)
    total_windows = 0

    # Loop through all batches in the dataloader
    for _, labels in dataloader:
        total_labels += labels.sum(dim=0)  # Sum the binary labels for each class
        total_windows += labels.size(0)    # Count total number of windows

    # Exclude the absence class from the class proportions
    class_proportions = total_labels / total_windows  # Proportion of each class in the dataset
    return class_proportions

def compute_chance_map(class_proportions):
    """
    Compute chance MAP based on class proportions (excluding absence class).

    Args:
        class_proportions (torch.Tensor): Proportion of each class.

    Returns:
        float: Chance MAP based on positive class proportions.
    """
    return (class_proportions / 2).mean().item()

# Main function to compute the chance baseline
def chance_baseline(dataloader):
    # Compute class proportions (excluding absence class)
    class_proportions = compute_class_proportions_from_dataloader(dataloader)
    
    # Compute chance MAP
    chance_map = compute_chance_map(class_proportions)
    print(f"Class Proportions: {class_proportions.numpy()}")
    print(f"Chance MAP: {chance_map:.4f}")

    # Compute MAP for the most frequent class
    map_most_frequent = np.max(class_proportions.numpy())
    print(f"Most frequent MAP : {map_most_frequent:.4f}")

    # Optionally, plot the class proportions for visualization
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(class_proportions)), class_proportions.numpy(), color='skyblue')
    plt.title('Class Proportions (Excluding No Label Class)')
    plt.xlabel('Class Index')
    plt.ylabel('Proportion')
    plt.xticks(range(len(class_proportions)), [f"Class {i}" for i in range(len(class_proportions))], rotation=90)
    plt.tight_layout()
    plt.savefig('class_proportions_no_label.png')

    return class_proportions.numpy()

# Example usage (assuming dataloader is defined):
# dataloader = your_dataloader_object
# chance_baseline(dataloader)
