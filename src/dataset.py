"""Dataset and label tools for plant disease classification.

Handles the extraction of disease-only labels from folder names
that contain both plant species and disease (e.g. "tomato late blight" to "late blight"),
and provides a PyTorch Dataset for loading and transforming the images.
"""

import os
from PIL import Image
from torch.utils.data import Dataset


def get_plant_prefix(folder, all_folders):
    """Find the longest prefix of this folder name shared with another folder.

    Used to identify the plant species portion of folder names. For example,
    if "tomato early blight" and "tomato late blight" both exist, "tomato"
    is detected as the shared prefix (plant name). Handles multi-word plant
    names like "bell pepper" automatically.

    Args:
        folder (str): Single folder name (e.g. "tomato late blight").
        all_folders (list[str]): All folder names in the dataset.

    Returns:
        str: The plant prefix (e.g. "tomato", "bell pepper").
            Falls back to the first word if no shared prefix is found.
    """
    words = folder.split()
    for n in range(len(words) - 1, 0, -1):
        prefix = " ".join(words[:n])
        if any(f != folder and f.startswith(prefix + " ") for f in all_folders):
            return prefix
    return words[0]


def extract_disease(folder, all_folders):
    """Extracts disease name from folder by removing the plant prefix.

    Args:
        folder (str): Folder name 
        all_folders (list[str])

    Returns:
        str: Disease name only
    """
    prefix = get_plant_prefix(folder, all_folders)
    return folder[len(prefix) + 1 :]


def build_label_map(data_dir):
    """Builds a mapping from disease name to integer label.

    Scans the data directory for subfolders, extracts disease names
    and assigns each unique disease a sequential integer index.

    Args:
        data_dir (str): Path to dataset split (e.g. "/content/data/train").

    Returns:
        tuple: (label_map, folders)
            - label_map (dict): Disease name to int index (e.g. {"late blight": 20}).
            - folders (list[str]): Sorted list of all subfolder names, used as
              reference for extract_disease() in other splits.
    """
    folders = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )
    diseases = sorted(set(extract_disease(f, folders) for f in folders))
    label_map = {d: i for i, d in enumerate(diseases)}
    return label_map, folders


class PlantDiseaseDataset(Dataset):
    """PyTorch dataset for plant disease images.

    Loads images from a directory structure where each subfolder contains
    images for one plant-disease combination. Maps each image to its
    disease-only label using the provided label_map.

    Args:
        data_dir (str): Path to dataset split
        label_map (dict): Disease name to int index from build_label_map().
        all_folders (list[str]): Reference folder list from build_label_map(),
            used to extract disease names consistently across splits.
        transform (callable, optional): Torchvision transforms to apply to
            each image (resize, augmentation, normalization).

    Attributes:
        samples (list[tuple]): List of (image_path, label) pairs.
    """

    def __init__(self, data_dir, label_map, all_folders, transform=None):
        self.transform = transform
        self.samples = []

        folders = [
            d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
        ]
        for folder in folders:
            disease = extract_disease(folder, all_folders)
            label = label_map[disease]
            folder_path = os.path.join(data_dir, folder)
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(folder_path, img_name), label))

    def __len__(self):
        """Returns total number of images in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Loads and returns a single (image, label) pair.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image, label)
                - image (Tensor or PIL.Image): Transformed image if transform is set.
                - label (int): Integer disease class index.
        """
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
