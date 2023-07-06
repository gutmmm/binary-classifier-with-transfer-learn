"""Training module for FashionMNIST binary classificator"""
import logging
import os

import torch
from torch import device, nn
from tqdm import tqdm

from common.tools import calculate_accuracy, plot_losses
from common.dataset_processing import Processor
from model import BinaryFMnist

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

dev = device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    """Training and validation loop"""
    epoch_loss, val_epoch_loss, val_epoch_accuracy = [], [], []
    for epoch in range(EPOCHS):
        running_loss = 0.0
        # Train
        for img, label in tqdm(train_dataloader, leave=True, desc=f"Epoch:{epoch+1}/{EPOCHS}"):
            img, label = img.to(dev), label.to(dev)
            output = model(img)
            output = output.squeeze()
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss.append(running_loss)

        # Validate
        model.eval()
        with torch.no_grad():
            val_running_loss = 0.0
            val_running_accuracy = 0.0

            for v_img, v_label in val_dataloader:
                v_img, v_label = v_img.to(dev), v_label.to(dev)
                output = model(v_img)
                output = output.squeeze()
                val_loss = criterion(output, v_label)
                val_running_loss += val_loss.item()

                val_accuracy = calculate_accuracy(output, v_label)
                val_running_accuracy += val_accuracy

            val_epoch_loss.append(val_running_loss)
            val_epoch_accuracy.append(val_running_accuracy / len(val_dataloader))

    logging.info("Training loss: %s", sum(epoch_loss) / len(epoch_loss))
    logging.info("Validation loss: %s", sum(val_epoch_loss) / len(val_epoch_loss))
    logging.info("Validation accuracy: %s ", sum(val_epoch_accuracy) * 100 / len(val_epoch_accuracy))
    plot_losses(epoch_loss, val_epoch_loss)


logging.info("Processing dataset")
process = Processor()
process.rebalance_and_relabel()
train_dataloader, val_dataloader, _ = process.split_database()
logging.info("Processing finished")

EPOCHS = 100
LEARNNING_RATE = 0.00001

model = BinaryFMnist().to(dev)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNNING_RATE)

train()

torch.save(model.state_dict(), "models/model.pth")
