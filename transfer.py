"""Module for transfer learning based on FashionMNIST binary classificator to distinct between flats and heels"""
import logging
import os

import torch
import torchvision
from torch import device, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from common.tools import calculate_accuracy
from model import BinaryFMnist

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

dev = device("cuda" if torch.cuda.is_available() else "cpu")


class ShoeDataset(Dataset):
    """Custom PyTorch dataset class"""

    def __init__(self, imgs, label):
        self.image = imgs
        self.label = label

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.image[idx]
        label = self.label[idx]
        return image, label


class TransferLearn:
    """Transfer learning pipeline"""

    def __init__(self):
        self.flats_train = torchvision.io.read_image("heels_flats/train/Flats/107.png")
        self.heels_train = torchvision.io.read_image("heels_flats/train/Heels/746.png")

        self.flats_test = [
            torchvision.io.read_image(os.path.join("./heels_flats/test/Flats", img_dir))
            for img_dir in os.listdir("./heels_flats/test/Flats")
        ]
        self.heels_test = [
            torchvision.io.read_image(os.path.join("./heels_flats/test/Heels", img_dir))
            for img_dir in os.listdir("./heels_flats/test/Heels")
        ]

        self.model = BinaryFMnist()
        self.model.load_state_dict(torch.load("models/model.pth"))

    def freeze_and_replace_layers(self):
        """Prepare model for transfer learning.
        Freeze 2 last leyers and replace with a new ones
        """

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace 2 last FC layers
        self.model.linear1 = nn.Linear(192, 120)
        self.model.linear2 = nn.Linear(120, 60)
        self.model.out = nn.Linear(60, 1)

        return self.model

    def prepare_train(self):
        """Prepare training examples"""

        img = [self.flats_train, self.heels_train]

        transform = transforms.Compose(
            [transforms.ConvertImageDtype(torch.float32), transforms.Normalize((0.5,), (0.5,))]
        )

        images = [transform(x) for x in img]
        label_flats, label_heels = torch.tensor(1.0), torch.tensor(0.0)
        labels = [label_flats, label_heels]

        train_dataset = ShoeDataset(images, labels)
        return DataLoader(train_dataset, batch_size=1, shuffle=True)

    def prepare_test(self):
        """Prepare testing examples"""

        transform = transforms.Compose(
            [transforms.ConvertImageDtype(torch.float32), transforms.Normalize((0.5,), (0.5,))]
        )

        flats_images = [transform(x) for x in self.flats_test]
        heels_images = [transform(x) for x in self.heels_test]

        flats_labels = [torch.tensor(1.0) for x in range(len(flats_images))]
        heels_labels = [torch.tensor(0.0) for x in range(len(heels_images))]

        images = flats_images + heels_images
        labels = flats_labels + heels_labels

        test_dataset = ShoeDataset(images, labels)
        return DataLoader(test_dataset, batch_size=1, shuffle=False)

    def retrain(self, data):
        """Retrain layers"""

        epoch_loss = []
        for epoch in range(EPOCHS):
            running_loss = 0.0
            for img, label in tqdm(data, leave=True, desc=f"Epoch:{epoch}/{EPOCHS}"):
                img, label = img.to(dev), label.to(dev)

                output = model(img)
                output = output[0]

                loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_loss.append(running_loss)

    def test(self, data):
        """Test the model"""

        test_loss = 0.0
        test_accuracy = 0.0
        model.eval()
        with torch.no_grad():
            for t_img, t_label in data:
                t_img, t_label = t_img.to(dev), t_label.to(dev)
                output = model(t_img)
                output = output[0]
                test_loss += criterion(output, t_label).item()  # Convert labels to Long data type
                test_accuracy += calculate_accuracy(output, t_label)

        test_loss /= len(test_data)
        test_accuracy /= len(test_data)

        print("Testing Loss:", test_loss)
        print(f"Testing Accuracy: {test_accuracy * 100}%")


EPOCHS = 50
LEARNNING_RATE = 0.001

tranfer_run = TransferLearn()
model = tranfer_run.freeze_and_replace_layers()
train_data = tranfer_run.prepare_train()
test_data = tranfer_run.prepare_test()

model = model.to(dev)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNNING_RATE)

tranfer_run.retrain(train_data)
tranfer_run.test(test_data)
