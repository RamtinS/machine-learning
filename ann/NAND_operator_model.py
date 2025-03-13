import matplotlib.pyplot as plt
import numpy as np
import torch

# The four possible input combinations for NAND.
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
  [1.0],
  [1.0],
  [1.0],
  [0.0]],
  dtype=torch.float32)

# The NAND operator returns only 0 when both the inputs are 1, and it returns 1 otherwise.
class NANDOperatorModel(torch.nn.Module):
  def __init__(self):
    super(NANDOperatorModel, self).__init__()
    self.W = torch.rand((2, 1), dtype=torch.float32, requires_grad=True) # 2x1 matrix with random numbers between 0 and 1.
    self.b = torch.rand((1, 1), dtype=torch.float32, requires_grad=True) # 1x1 matrix with random numbers between 0 and 1.

  def f (self, x):
    return torch.sigmoid(self.logits(x))

  def logits(self, x):
      return x @ self.W + self.b

  def loss(self, x, y):
    loss_fn = torch.nn.BCEWithLogitsLoss()
    return loss_fn(self.logits(x), y)

model = NANDOperatorModel()

optimizer = torch.optim.SGD([model.W, model.b], lr=0.1)
for epoch in range(20000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W.data.numpy()[0], model.b.item(), model.loss(x_train, y_train).item()))

# Creates a grid.
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)  # Create a gird of x and y values.

# Flatten the grid to input it to the model
X_flat = np.c_[X.ravel(), Y.ravel()]
X_flat_tensor = torch.tensor(X_flat, dtype=torch.float32)

# Calculate the model's output
Z = model.f(X_flat_tensor).detach().numpy().reshape(X.shape)

# Create a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D surface on the graph.
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# Plot the training points.
ax.scatter(x_train[:, 0], x_train[:, 1], y_train[:, 0], color='red', s=100, alpha=1.0, label='Training Points')

# Labels and titles
ax.set_xlabel('Input 1')
ax.set_ylabel('Input 2')
ax.set_zlabel('Output')
plt.title('NAND Operator')

plt.legend()
plt.show()