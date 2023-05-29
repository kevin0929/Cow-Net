import torch
from Pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Optional


class CowDataset(Dataset):
    def __init__(self, img_path: Path, features_path: Path):
        """_summary_

        Args:
            features(list): The information of this cow (coordinate, bb_size)
            img_paths (Path): The image path of this cow
            labels (list): The behavior class of this cow
            transformer (function): What we want to do for the image.
        """
        self.img_path = img_path
        self.features_path = features_path
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # The size of ResNet except to input
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        # load feature info and convert it to tensor
        features_path = self.features_path[idx]
        f = open(features_path, "r")
        features_string = f.read()

        string_info_list = features_string.split(",")
        int_info_list = []
        for string in string_info_list:
            int_info_list.append(float(string))
        features = torch.tensor(int_info_list, dtype=torch.float32)

        # load image and apply transformation
        img_path = self.img_path[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # depend on img name to pick label
        label = img_path.split("/")[-1].split("_")[0]

        return image, features, label
