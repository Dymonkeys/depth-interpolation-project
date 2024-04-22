from torch.utils.data import Dataset

class DepthDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx][0]
        depth_map = self.data[idx][1]

        if self.transform:
            image = self.transform(image)
            depth_map = self.transform(depth_map)

        return image, depth_map
    


"""
class DepthDataset(Dataset):
    def __init__(self, images, depth_maps, transform=None):
        self.images = images
        self.depth_maps = depth_maps
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        depth_map = self.depth_maps[idx]

        if self.transform:
            image = self.transform(image)
            depth_map = self.transform(depth_map)

        return image, depth_map
"""


"""
import os
from torch.utils.data import Dataset
from PIL import Image

class DepthDataset(Dataset):
    def __init__(self, image_dir, depth_dir, transform=None):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.transform = transform

        # Get the list of image filenames
        self.image_filenames = os.listdir(image_dir)
        self.depth_filenames = os.listdir(depth_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(image_path).convert('RGB')

        # Load depth map
        depth_path = os.path.join(self.depth_dir, self.depth_filenames[idx])
        depth_map = Image.open(depth_path).convert('L')

        if self.transform:
            image = self.transform(image)
            depth_map = self.transform(depth_map)

        return image, depth_map
"""