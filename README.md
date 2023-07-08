# Binary image classificator with transfer learning

This repository contains an implementation of a binary classifier based on a deep neural network using PyTorch. The classifier is trained on the FashionMNIST dataset to distinguish between shoes and other types of clothes. Additionally, it includes a component of transfer learning to distinguish between two new types of shoes.

Dataset
The dataset used for training and evaluation is the FashionMNIST dataset. FashionMNIST is a collection of 70,000 grayscale images of 10 different fashion categories, each represented by 28x28 pixels. In this project, we focus on distinguishing between shoes and other clothing items.

### Requirements

You can install the necessary dependencies using pip:


`pip install -r requirements.txt` or `conda env create -f requirements.yml`


### Model Architecture and pipeline
The binary classifier model architecture is defined in the BinaryFMnist class in the model.py file. 

Usage
To train the binary classifier, execute the following command:

`python train.py`

This script will train the deep neural network using the FashionMNIST dataset. The trained model will be saved to the disk.

To evaluate the classifier on the test set, run the following command:

`python test.py`

This script will load the trained model and evaluate its performance on the test set. It will output the accuracy and other evaluation metrics.

After training the initial binary classifier, transfer learning is performed to distinguish between two new types of shoes. To execute the transfer learning process, run the following command:

`python transfer.py`
