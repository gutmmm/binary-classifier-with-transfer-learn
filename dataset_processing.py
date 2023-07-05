import os
import numpy as np
from tqdm import tqdm
from torchinfo import summary
import matplotlib.pyplot as plt


import torch
import torchvision
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms, ToTensor




class processor():
    def __init__(self, split_ratio: float = 0.8, batch_size:int = 128):
        self.SPLIT_RATIO = split_ratio
        self.BATCH_SIZE = batch_size

    def load_and_transform(self):
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
        self.train_dataset = FashionMNIST(root='./', train=True, download=True, transform=transform)
        self.test_dataset =  FashionMNIST(root='./', train=False, download=True, transform=transform)

    def rebalance_and_relabel(self):
        # Rebalancing dataset -> 18k shoes + 7 other classes x 2571 ~= 18k
        shoes, other = [], []

        for idx, element in enumerate(self.train_dataset):
            if element[1] == 5 or element[1] == 7 or element[1] == 9:
                shoes.append(idx)
            else:
                other.append(idx)

        shoes_data = [self.train_dataset[x] for x in shoes]

        items_dict = self.train_dataset.class_to_idx
        clothes = []
        for item in items_dict:
            if items_dict[item] != 5 and items_dict[item] != 7 != 9:
                x = [self.train_dataset[x] for x in other if self.train_dataset[x][1] == items_dict[item]][:2571] #2571 examples from each class of non shoe category.
                clothes.extend(x)

        relabeled_shoe_data = []
        for data in shoes_data:
            new_data = list(data)
            new_data[1] = 1.0
            relabeled_shoe_data.append(new_data)

        relabeled_clothes_data = []
        for data in clothes:
            new_data = list(data)
            new_data[1] = 0.0
            relabeled_clothes_data.append(new_data)

        self.train_set = relabeled_shoe_data + relabeled_clothes_data

        # Test data
        self.relabeled_test_data = []
        for data in self.test_dataset:
            if data[1] == 5 or data[1] == 7 or data[1] == 9:
                new_data = list(data)
                new_data[1] = 1.0
                self.relabeled_test_data.append(new_data)
            else:
                new_data = list(data)
                new_data[1] = 0.0
                self.relabeled_test_data.append(new_data)


    def split_database(self):

        train_size = int(len(self.train_set)*self.SPLIT_RATIO)
        val_size = int(len(self.train_set) - train_size)

        train, validation = random_split(dataset = self.train_set,
                                lengths = [train_size, val_size],
                                generator = torch.Generator().manual_seed(77))

        train_dataloader = DataLoader(dataset=train, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=2)
        val_dataloader = DataLoader(dataset=validation, batch_size=self.BATCH_SIZE, shuffle = False, num_workers=2)
        test_dataloader = DataLoader(dataset=self.relabeled_test_data, batch_size=self.BATCH_SIZE, shuffle = False, num_workers=2)

        return train_dataloader, val_dataloader, test_dataloader


