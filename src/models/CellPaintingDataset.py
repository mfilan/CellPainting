import numpy as np
import pandas as pd
import torch
from tifffile import imread

from src.config import files


class CellPaintingDataset(torch.utils.data.Dataset):
    def __init__(self, data, annotation_file, transform=None):
        self.data = data
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img = imread(self.data / self.annotations.iloc[index, 0]).astype(np.float32)
        y_label = self.annotations.loc[index, 'compound_label']
        if self.transform is not None:
            img = self.transform(img)
        return img, y_label


if __name__ == "__main__":
    x = CellPaintingDataset(files.data_processed, files.data_annotations)
    print(x.__getitem__(0))