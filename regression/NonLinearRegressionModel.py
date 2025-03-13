import torch
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x): # Implementation of sigmoid function sig(x) = 1 / (1+e^(-x))
  return 1 / (1 + torch.exp(-x))

data = pd.read_csv('day_head_circumference.csv')

x_train = torch.tensor(data[['day']].values, dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(data[['head circumference']].values, dtype=torch.float32).reshape(-1, 1)

class NonLinearRegressionModel:
  def __init__(self):
    self.W = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
    self.b = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)

  def f(self, x):
    return 20 * sigmoid(x @ self.W + self.b) + 31 # Non linear model: f(x) = 20 * sig(xW + b) + 31

  def loss(self, x, y):
    return torch.mean(torch.square(self.f(x) - y))

model = NonLinearRegressionModel()

optimizer = torch.optim.SGD([model.W, model.b], lr=0.0000001)
for epoch in range(100000):
  model.loss(x_train, y_train).backward()
  optimizer.step()
  optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W.item(), model.b.item(), model.loss(x_train, y_train).item()))

# Show result.
plt.plot(x_train, y_train, 'o', label='Observations')
plt.xlabel('Age (days)')
plt.ylabel('Head Circumference')
x_range = torch.linspace(torch.min(x_train), torch.max(x_train), steps=100).reshape(-1, 1)
plt.plot(x_range, model.f(x_range).detach(), label='f(x) = 20 * sigmoid(xW + b) + 31', color='orange') # Plot the model's predictions over range x.
plt.legend()
plt.show()

