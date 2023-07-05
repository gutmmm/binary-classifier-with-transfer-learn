import os
import logging

import torch
from torch import nn

import common
from model import BinaryFMnist
from dataset_processing import processor



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.BCEWithLogitsLoss()
model = BinaryFMnist()
model.load_state_dict(torch.load('./model.pth'))
model.eval()

logging.info('Processing dataset')
process = processor()
process.load_and_transform()
process.rebalance_and_relabel()
_, _, test_dataloader = process.split_database()
logging.info('Processing finished')

with torch.no_grad():
    test_loss = 0.0
    test_accuracy = 0.0
    for t_img, t_label in test_dataloader:
        t_img, t_label = t_img.to(device), t_label.to(device)
        output = model(t_img)
        output = output.squeeze()
        test_loss += criterion(output, t_label).item()  # Convert labels to Long data type
        test_accuracy += common.calculate_accuracy(output, t_label)

test_loss /= len(test_dataloader)
test_accuracy /= len(test_dataloader)

logging.info('Testing Loss:', test_loss)
logging.info(f'Testing Accuracy: {test_accuracy * 100}%')