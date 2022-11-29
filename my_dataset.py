from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
	def __init__(self, root, transform=None):
		self.root = root
		self.img_list = self.get_img_list(root)
		self.len = len(self.img_list)
		self.transform = transform

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		img_path = self.img_list[index]
		img_f_path = os.path.join(self.root, img_path)
		img = Image.open(img_f_path).convert("RGB")
		if self.transform is not None:
			img = self.transform(img)
		return img
	
	def get_img_list(self, path):
		IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'JPEG', 'PNG', 'JPG'  # include image suffixes
		list_img = [img for img in os.listdir(path) if (img.split(".")[-1] in IMG_FORMATS) ==True]
		return list_img

#==========================================================================
if __name__ == "__main__":

	# Demo usage of the code
	import matplotlib.pyplot as plt
	from torchvision import transforms 
	
	data_transforms = transforms.Compose([
			transforms.Resize(64),
			transforms.CenterCrop(64),
		])

	DATASET_FOLDER = "catsfaces_64x64"

	my_dataset = MyDataset(root=DATASET_FOLDER, transform=data_transforms)
	print("my_dataset.__len__ : ", my_dataset.__len__())
	
	# Plots some samples from the dataset
	def show_images(dataset, num_samples=20, cols=5):
		plt.figure(figsize=(5,5)) 
		for i, img in enumerate(dataset):
			if i == num_samples:
				break
			plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
			plt.imshow(img) # imshow for PIL image
			plt.axis('off')
		plt.show()
	show_images(my_dataset)