import os
import logging
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

from model import BinaryFMnist
from dataset_processing import processor
import common



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

def train():
    epoch_loss, val_epoch_loss, val_epoch_accuracy = [], [], []
    for epoch in range(EPOCHS):
        running_loss = 0.0
        #Train
        for img, label in tqdm(train_dataloader, leave=True, desc=f'Epoch:{epoch+1}/{EPOCHS}'):
            img, label = img.to(device), label.to(device)
            output = model(img)
            output = output.squeeze()
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss.append(running_loss)

        #Validate
        model.eval()
        with torch.no_grad():
            val_running_loss = 0.0
            val_running_accuracy = 0.0

            for v_img, v_label in val_dataloader:
                v_img, v_label = v_img.to(device), v_label.to(device)
                output = model(v_img)
                output = output.squeeze()
                val_loss = criterion(output, v_label)
                val_running_loss += val_loss.item()

                val_accuracy = calculate_accuracy(output, v_label)
                val_running_accuracy += val_accuracy

            val_epoch_loss.append(val_running_loss)
            val_epoch_accuracy.append(val_running_accuracy / len(val_dataloader))
            
    logging.info("Training loss: ", sum(epoch_loss)/len(epoch_loss))
    logging.info("Validation loss: ", sum(val_epoch_loss)/len(val_epoch_loss))
    logging.info("Validation accuracy: ", sum(val_epoch_accuracy)*100/len(val_epoch_accuracy))
    common.plot_losses(epoch_loss, val_epoch_loss)



logging.info('Processing dataset')
process = processor()
process.load_and_transform()
process.rebalance_and_relabel()
train_dataloader, val_dataloader, _ = process.split_database()
logging.info('Processing finished')


EPOCHS = 10
LEARNNING_RATE = 0.00001

model = BinaryFMnist().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNNING_RATE)

train()

torch.save(model.state_dict(), './model.pth')