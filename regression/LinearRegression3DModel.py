import torch
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('day_length_weight.csv')

# Prepare training data.
x_length = torch.tensor(data['length'].values, dtype=torch.float32).reshape(-1, 1)
x_weight = torch.tensor(data['weight'].values, dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(data['day'].values, dtype=torch.float32).reshape(-1, 1)

# Combine the length and the weight to one tensor so that the model
# can use data from both simultaneously to predict age.
x_train = torch.cat((x_length, x_weight), dim=1)

class LinearRegression3DModel:
  def __init__(self):
    self.W = torch.tensor([[0.0], [0.0]], dtype=torch.float32, requires_grad=True)
    self.b = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)

  # Predict age
  def f(self, x):
    return x @ self.W + self.b

  def loss(self, x, y):
    return torch.mean(torch.square(self.f(x) - y))

model = LinearRegression3DModel()

optimizer = torch.optim.SGD([model.W, model.b], lr=0.000115)
for epoch in range(100000):
  model.loss(x_train, y_train).backward()
  optimizer.step()
  optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W.data.numpy(), model.b.item(), model.loss(x_train, y_train).item()))



# -- Show the result in a 3D graph --
fig = plt.figure() # Create a new figure
ax = fig.add_subplot(111, projection='3d') # 3D plots require a subplot with 3D projection enabled

# Plot the observed data.
ax.scatter(x_length, x_weight, y_train, c='r', marker='o', label='Observations')

# Create a grid to plot the regression plane
length_range = torch.linspace(x_train[:, 0].min(), x_train[:, 0].max(), 100)
weight_range = torch.linspace(x_train[:, 1].min(), x_train[:, 1].max(), 100)

# Create a grid of length and weight values.
length_grid, weight_grid = torch.meshgrid(length_range, weight_range, indexing='ij')

# Flatten the grid and stack the length and weight values to create a 2D tensor of grid points
grid = torch.stack([length_grid.flatten(), weight_grid.flatten()], dim=1)

# Predict day values for the grid points
day_pred = model.f(grid).detach().numpy().reshape(length_grid.shape)

# Plot the regression plane
ax.plot_surface(length_grid.numpy(), weight_grid.numpy(), day_pred, color='b', alpha=0.5, rstride=100, cstride=100, label='Regression Plane')

# Set labels and show the plot
ax.set_xlabel('Length')
ax.set_ylabel('Weight')
ax.set_zlabel('Day')
plt.legend()
plt.show()

