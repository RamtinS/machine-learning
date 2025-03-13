import torch
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('length_weight.csv') # Read data from the CSV file using the Pandas library.

# Prepare the training data.
# Convert the length column to a tensor matrix with a single column.
x_train = torch.tensor(data[['length']].values, dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(data[['weight']].values, dtype=torch.float32).reshape(-1, 1)

class LinearRegressionModel:
  def __init__(self): # Constructor for the class.
    self.W = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True) # Create a tensor (1x1 matrix in this case) that starts from 0.0
    self.b = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True) # Create a tensor (1x1 matrix) for the bias, starting from 0.0

  # Predict the value.
  def f(self, x):
    # Multiply matrix x with matrix W and add matrix b
    return x @ self.W + self.b # Linear model: f(x) = xW + b

  # Compute Mean Squared Error (MSE) between predicted and actual values
  def loss(self, x, y):
    return torch.mean(torch.square(self.f(x) - y))

model = LinearRegressionModel() # Crests an object of the class.

# Train the model.
optimizer = torch.optim.SGD([model.W, model.b], lr=0.0001) # Lower the learning rate (lr) to prevent to numerical issues.
for epoch in range(100000): # Adjust range to reduce loss.
  loss = model.loss(x_train, y_train)
  loss.backward()# Calculate the loss. Compute gradients backwards.
  optimizer.step() # Adjust W and b to reduce loss.
  optimizer.zero_grad() # Clear the gradients for the next iteration
  if epoch % 100 == 0:
    print(loss.item())

# Print out data to evaluate the model.
print("W = %s, b = %s, loss = %s" % (model.W.item(), model.b.item(), model.loss(x_train, y_train).item()))

# Show result.
plt.plot(x_train, y_train, 'o', label='Observations')
plt.xlabel('Length')
plt.ylabel('Weight')
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]], dtype=torch.float32)
plt.plot(x, model.f(x).detach(), label='f(x) = xW+b')
plt.legend()
plt.show()