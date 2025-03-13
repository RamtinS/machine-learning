import ssl
import certifi
import urllib.request

ssl_context = ssl.create_default_context(cafile=certifi.where())
urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context)))

import matplotlib.pyplot as plt
import torch
import torchvision

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

# The images in the dataset are 28x28 pixels.
# Therefore, each image can be flattened into a vector of 784 features (28*28).
input_features = 784

# The total number of distinct categories the model is trying to classify. In this case, digits from 0 to 9.
number_of_classes = 10

class SoftmaxClassifier(torch.nn.Module):
  def __init__(self):
    super(SoftmaxClassifier, self).__init__()
    # W is a 784x10 Tensor with values centered around 0 and then scaled by 0.1.
    self.W = torch.nn.Parameter(torch.randn(input_features, number_of_classes) * 0.1, requires_grad=True)
    # b is a 1 dim tensor with a length of 10 containing all zeros.
    self.b = torch.nn.Parameter(torch.zeros(number_of_classes), requires_grad=True)

  # Softmax will give a probability between 0 and 1 for each element in the output.
  def f(self, x):
    return torch.softmax(self.logits(x), dim=1)

  def logits(self, x):
      return x @ self.W + self.b

  def accuracy(self, x, y):
    return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

  def loss(self, x, y):
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn(self.logits(x), y)

model = SoftmaxClassifier()

optimizer = torch.optim.SGD([model.W, model.b], lr=0.1)
for epoch in range(1000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s, accuracy = %s" % (model.W.detach().numpy(), model.b.detach().numpy(),
                                                    model.loss(x_train, y_train).item(),
                                                    model.accuracy(x_test, y_test).item()))

fig = plt.figure("Digits")

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(model.W[:, i].detach().numpy().reshape(28, 28))
    plt.title(f'W: {i}')
    plt.xticks([])
    plt.yticks([])

plt.show()