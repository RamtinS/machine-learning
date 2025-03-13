import matplotlib.pyplot as plt
import numpy as np
import torch

# The four possible input combinations for XOR.
# Input as a tensor shaped like a 4x2 matrix.
x_train = torch.tensor([
  [0.0, 0.0],
  [0.0, 1.0],
  [1.0, 0.0],
  [1.0, 1.0]],
  dtype=torch.float32)

# The four expected outputs for the given input.
# Output as a tensor shaped like a 4x1 matrix.
y_train = torch.tensor([
  [0.0],
  [1.0],
  [1.0],
  [0.0]],
  dtype=torch.float32)

# The XOR operator returns 1 exactly one of the inputs are 1, and 0 if both inputs are the same.
class XOROperatorModel(torch.nn.Module):
  def __init__(self):
    super(XOROperatorModel, self).__init__()
    # rand() creates a tensor with random number between [0, 1).
    # By multiplying the function with 2 and subtract by 1, we move the interval to (-1, 1).
    self.W1 = torch.nn.Parameter(torch.rand((2, 2)) * 2 - 1, requires_grad=True)  # Tensor as a 2x2 matrix
    self.b1 = torch.nn.Parameter(torch.rand((1, 2)) * 2 - 1, requires_grad=True)  # Tensor as a 1x2 matrix
    self.W2 = torch.nn.Parameter(torch.rand((2, 1)) * 2 - 1, requires_grad=True)  # Tensor as a 2x1 matrix
    self.b2 = torch.nn.Parameter(torch.rand((1, 1)) * 2 - 1, requires_grad=True)  # Tensor as a 1x1 matrix

  # When working with the XOR operator, we need to use multiple layers. This is because the XOR problem is not
  # linearly separable. This means you can't draw a single straight line to separate the 0 outputs from the 1 outputs.

  # Layer 1 (first hidden layer)
  def f1(self, x):
    return torch.sigmoid(x @ self.W1 + self.b1)

  # Layer 2 (output layer)
  def f2(self, x):
    return torch.sigmoid(x @ self.W2 + self.b2)

  # The second layer takes the output from the first layer and produces the final output.
  def f(self, x):
    return self.f2(self.f1(x))

  def loss(self, x, y):
    loss_fn = torch.nn.BCELoss() # Binary cross entropy
    return loss_fn(self.f(x), y)

model = XOROperatorModel()

optimizer = torch.optim.SGD([model.W1, model.b1, model.W2, model.b2], lr=0.1)
for epoch in range(200000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

print("W1 = %s,\nW2 = %s,\nb1 = %s,\nb2 = %s,\nloss = %s" % (model.W1.detach().numpy(), model.W2.detach().numpy(),
                                                         model.b1.detach().numpy(), model.b2.detach().numpy(),
                                                         model.loss(x_train, y_train).item()))

# Creates a grid.
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y) # Create a gird of x and y values.

# Flatten the grid to input it to the model
X_flat = np.c_[X.ravel(), Y.ravel()]
X_flat_tensor = torch.tensor(X_flat, dtype=torch.float32)

# Calculate the model's output
Z = model.f(X_flat_tensor).detach().numpy().reshape(X.shape)

# Create a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D surface of the graph
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# Plot the training points.
ax.scatter(x_train[:, 0], x_train[:, 1], y_train[:, 0], color='red', s=100, alpha=1.0, label='Training Points')

# Set the labels and title
ax.set_xlabel('Input 1')
ax.set_ylabel('Input 2')
ax.set_zlabel('Output')
plt.title('XOR Operator')

plt.legend()
plt.show()