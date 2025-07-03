
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TuringPatternDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the metadata.csv file
            image_dir (str): Directory with all the pattern images
            transform (callable, optional): Optional transform to be applied on an image sample
        """
        self.image_dir = image_dir
        self.transform = transform if transform else transforms.ToTensor()

        # Load metadata with 9 fields (including image_saved)
        self.metadata = pd.read_csv(csv_file)

        # Keep only rows where image was successfully saved (image_saved == 1)
        self.metadata = self.metadata[self.metadata['image_saved'] == 1]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])

        # Open and convert image to grayscale tensor
        image = Image.open(img_path).convert('L')
        image = self.transform(image)

        # Extract and return the 4 parameters
        params = torch.tensor([
            row['D_u (mm^2/s)'],
            row['D_v (mm^2/s)'],
            row['F (1/s)'],
            row['k (1/s)']
        ], dtype=torch.float32)

        return image, params



# Returns (image_tensor, parameter_tensor) for training

# Only loads images where image_saved == 1

# Works with your metadata.csv and patterns/ folder