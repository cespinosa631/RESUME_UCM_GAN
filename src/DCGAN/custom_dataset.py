import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_size):
        self.img_labels = pd.read_csv(annotations_file)
        self.resize_transform = transforms.Resize((img_size, img_size))
        self.transform_tensor = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 1]
        image = Image.open(img_path)

        image = self.resize_transform(image)
        ori_image = self.transform_tensor(image)

        label = self.img_labels.iloc[idx, 2]
        return ori_image, label