import os
import torch
import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset

class roadDataset(VisionDataset):
    def __init__(self, root = None, transforms = None, transform = None, target_transform = None):
        super(roadDataset, self).__init__(root, transforms, transform, target_transform)
        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "targets")
        self.image_filenames = sorted(os.listdir(self.images_dir))
        self.mask_filenames = sorted(os.listdir(self.masks_dir))
        assert len(self.image_filenames) == len(self.mask_filenames)

    def __len__(self):
        return len(self.mask_filenames)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.images_dir, self.image_filenames[index])
        mask_path = os.path.join(self.masks_dir, self.mask_filenames[index])

        image = Image.open(image_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                mask = self.target_transform(mask)

        return image, mask

class inferenceDataset(Dataset):
    def __init__(self, input_dir = None, transform = None):
        self.input_dir = input_dir
        self.image_filenames = sorted(os.listdir(self.input_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, index):
        file_name = self.image_filenames[index]
        image_path = os.path.join(self.input_dir, file_name)

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        else:
            # 기본적으로 numpy array로 변환 후 tensor로 변환 (정규화: [0,1])
            image = np.array(image)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, file_name
        