import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
import tifffile as tiff
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



class BasicDataset(Dataset):
    def __init__(self, dir_img, dir_mask, transform=None):
        self.dir_img=dir_img
        self.dir_mask=dir_mask
        self.transform = transform

        self.input = os.listdir(dir_img)
        self.output = os.listdir(dir_mask)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input_name=self.input[idx]
        output_name=self.output[idx]
        input_img=Image.fromarray(tiff.imread(os.path.join(self.dir_img, input_name)))
        output_img=Image.fromarray(tiff.imread(os.path.join(self.dir_mask, output_name)))
        if self.transform is not None:
            input_img = self.transform(input_img)
            output_img = self.transform(output_img)
        return input_img, output_img


def Create_dataloader(dir_img, dir_mask, val_percent, batch_size=8, transform=None):

    dataset = BasicDataset(dir_img, dir_mask, transform=transform)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val= random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                            drop_last=True)

    return train_loader, val_loader




