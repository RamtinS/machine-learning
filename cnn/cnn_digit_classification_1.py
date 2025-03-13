import ssl
import certifi
import urllib.request

ssl_context = ssl.create_default_context(cafile=certifi.where())
urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context)))

import torch
import torch.nn as nn
import torchvision

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('./data/digit', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data/digit', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

# Normalization of inputs.
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Divide training data into batches to speed up optimization.
batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)

class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        # Model layers (includes initialized model variables):

        # First convolutional layer: Takes 1 input channel (image) and returns 32 output channels (feature maps).
        # It applies 32 filters (kernels) of size 5x5 to the input image. Each filter slides across the image and computes
        # the dot product between the filter and a local region of the input, detecting patterns such as edges or textures.
        # The more filters we add, the greater the model's capacity to learn a variety of features, but this increases
        # the computational cost and makes the model slower to train.
        # Padding of 2 ensures that the spatial dimensions of the output feature maps remain 28x28 pixels, matching the input size.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)

        # First pooling layer.
        # This layer takes 32 feature maps of size 28x28 and applies a 2x2 max pooling operation to each map.
        # Max pooling reduces the size of each feature map to 14x14 by taking the maximum value from each 2x2 patch.
        # The result is 32 feature maps, each with dimensions 14x14.
        # Pooling helps make the feature maps smaller and more manageable for the network while preserving the most
        # important features, making it easier for the model to process.
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Second convolution layer: Takes 32 input channels and returns 64 output channels (feature maps).
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)

        # Second pooling layer: Reduces the 64x14x14 feature maps to 64x7x7.
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected (dense) layer.
        # After the second pooling layer, the feature maps are reduced to 64 maps of size 7x7.
        # Before passing this data to the dense layer, the 3D tensor needs to be flattened into a 2D tensor
        # of shape (batch_size, 64 * 7 * 7), or (batch_size, 3136). Each input example becomes a 1D vector of size 3136.
        # This dense layer takes these flattened vectors as input, with 3136 features, and outputs 10 values per example.
        # The output size of 10 corresponds to the 10 classes in the MNIST dataset (digits 0-9).
        # Each output value represents the raw score for each digit, which will be used in the softmax function
        # to predict the probability of each class.
        self.dense = nn.Linear(64 * 7 * 7, 10)

    def logits(self, x):
        # Pass through the layers.
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.reshape(-1, 64 * 7 * 7) # Flatten the tensor.
        return self.dense(x)

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross-Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

model = ConvolutionalNeuralNetworkModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), 0.001)
for epoch in range(20):
    for batch in range(len(x_train_batches)):
        model.loss(x_train_batches[batch], y_train_batches[batch]).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b,
        optimizer.zero_grad()  # Clear gradients for next step

    print("accuracy = %s" % model.accuracy(x_test, y_test).item())


# accuracy = 0.9729999899864197
# accuracy = 0.9811000227928162
# accuracy = 0.9843000173568726