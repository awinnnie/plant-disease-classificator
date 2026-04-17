"""Dataset and label tools for plant disease classification."""

import os
from PIL import Image
from torch.utils.data import Dataset


def get_plant_prefix(folder, all_folders):
    """Find the longest prefix of this folder name shared with another folder (to seperate plants)."""
    words = folder.split()
    for n in range(len(words) - 1, 0, -1):
        prefix = ' '.join(words[:n])
        if any(f != folder and f.startswith(prefix + ' ') for f in all_folders):
            return prefix
    return words[0]


def extract_disease(folder, all_folders):
    """Extract disease name from folder by removing plant prefix."""
    prefix = get_plant_prefix(folder, all_folders)
    return folder[len(prefix) + 1:]


def build_label_map(data_dir):
    """Mapping from disease name to integer label."""
    folders = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    diseases = sorted(set(extract_disease(f, folders) for f in folders))
    label_map = {d: i for i, d in enumerate(diseases)}
    return label_map, folders


class PlantDiseaseDataset(Dataset):
    """a PyTorch dataset for plant disease images."""

    def __init__(self, data_dir, label_map, all_folders, transform=None):
        self.transform = transform
        self.samples = []

        folders = [
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ]
        for folder in folders:
            disease = extract_disease(folder, all_folders)
            label = label_map[disease]
            folder_path = os.path.join(data_dir, folder)
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(folder_path, img_name), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label