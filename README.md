# AssigmentM3

# Fashion MNIST Classifier

This project implements a neural network using TensorFlow and Keras to classify images in the Fashion MNIST dataset into 10 categories, such as T-shirts, trousers, and sneakers. The model is trained, validated, and evaluated, and training/validation accuracy and loss metrics are visualized.

# Dataset Overview

The Fashion MNIST dataset is a collection of 28x28 grayscale images of clothing items, categorized into 10 classes:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

The dataset contains:

- 60,000 training images
- 10,000 testing images

# Model Architecture

The neural network is built using Keras' Sequential API with the following architecture:

Input Layer:
- Flattens 28x28 images into a 784-dimensional vector.
- Hidden Layers:
- Layer 1: 128 neurons with ReLU activation.
- Layer 2: 64 neurons with ReLU activation.

Output Layer:

- 10 neurons with Softmax activation (for multi-class classification).
- Loss Function: Categorical Cross-Entropy.
- Optimizer: Adam.
- Metrics: Accuracy.

# Hyperparameters

- Number of Epochs: 10
- Batch Size: 32 (default)
- Validation Split: 20% of the training data
- Activation Functions:
- ReLU for hidden layers to handle non-linearities.
- Softmax for the output layer to convert predictions to probabilities.

# Results

- Training and validation accuracy/loss metrics are plotted after training.
- The model achieves reasonable accuracy on the validation set with potential for further tuning.

# Installation and Setup

## Prerequisites
1. Install Python 3.x (check using python3 --version).

2. Install required libraries:
```bash
pip install tensorflow matplotlib
```

## Running the Code
1. Clone the repository:
```bash
git clone https://github.com/YourUsername/YourRepository.git
cd YourRepository
```
2. Run the script to train and evaluate the model:
```bash
python3 fashion_mnist_classifier.py
```
3. Training and validation accuracy/loss plots will be displayed




