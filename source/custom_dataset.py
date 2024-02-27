from torch.utils.data import Dataset
from utils import get_unique_targets, square_crop
import cv2
import os


class CustomDataset(Dataset):
    def __init__(self, images_folder, labels_folder, transform=None):
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.image_names = os.listdir(images_folder)
        self.transform = transform
        self.unique_targets = get_unique_targets(labels_folder)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.images_folder, image_name)
        label_path = os.path.join(self.labels_folder,
                                  image_name.split(".")[0]+".txt")

        # Read the label
        with open(label_path, 'r') as f:
            label = f.read()

        # Read image using OpenCV
        image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply square crop and resize transformation
        image_tensor = square_crop(image_array, self.transform)

        # Normalize image pixel values
        image_tensor /= 255.0

        # Add channel dimension for grayscale image
        image_tensor = image_tensor.unsqueeze(0)

        return (image_tensor, self.unique_targets.index(label.strip()))
