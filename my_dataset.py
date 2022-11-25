from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.images = os.listdir(root)
        self.len = len(self.images)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img = self.images[index]
        path = os.path.join(self.root, img)
        img = np.array(Image.open(path).convert("RGB"))
        img = self.transform(img)
        return img