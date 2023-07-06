"""Testing module for FashionMNIST binary classificator"""
import logging
import os

import torch
from torch import nn

from common.tools import calculate_accuracy
from common.dataset_processing import Processor
from model import BinaryFMnist

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.BCEWithLogitsLoss()
model = BinaryFMnist()
model.load_state_dict(torch.load("models/model.pth"))
model.eval()

logging.info("Processing dataset")
process = Processor()
process.rebalance_and_relabel()
_, _, test_dataloader = process.split_database()
logging.info("Processing finished")

with torch.no_grad():
    TEST_LOSS = 0.0
    TEST_ACCURACY = 0.0
    for t_img, t_label in test_dataloader:
        t_img, t_label = t_img.to(device), t_label.to(device)
        output = model(t_img)
        output = output.squeeze()
        TEST_LOSS += criterion(output, t_label).item()  # Convert labels to Long data type
        TEST_ACCURACY += calculate_accuracy(output, t_label)

TEST_LOSS /= len(test_dataloader)
TEST_LOSS = f"{TEST_LOSS:.5f}"
TEST_ACCURACY /= len(test_dataloader)
TEST_ACCURACY = f"{TEST_ACCURACY*100:.2f}%"

logging.info("Testing Loss: %s", TEST_LOSS)
logging.info("Testing Accuracy: %s", TEST_ACCURACY)
