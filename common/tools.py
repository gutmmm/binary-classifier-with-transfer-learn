import matplotlib.pyplot as plt
import torch


def calculate_accuracy(outputs, labels):
    predictions = torch.round(torch.sigmoid(outputs))
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy


def plot_losses(epoch_loss, val_epoch_loss):
    fig, ax = plt.subplots()
    ax.plot(epoch_loss, label="Training")
    ax.plot(val_epoch_loss, label="Validation")
    ax.legend()
    plt.show()
