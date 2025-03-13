import torch
import matplotlib.pyplot as plt

# Two possible inputs for the NOT operator.
# Input as a nx1 matrix, where n is the number of rows (in this case two).
x_train = torch.tensor([1.0, 0.0], dtype=torch.float32).reshape(-1, 1)

# Expected output from the model for the given inout.
y_train = torch.tensor([0.0, 1.0], dtype=torch.float32).reshape(-1, 1)

# The not operator returns the opposite of the input. NOT(1) = 0.
class NOTOperatorModel(torch.nn.Module):
  def __init__(self):
    super(NOTOperatorModel, self).__init__()
    # rand() returns a 1x1 tensor filled with one random number between 0 and 1.
    self.W = torch.rand((1, 1), dtype=torch.float32, requires_grad=True)
    self.b = torch.rand((1, 1), dtype=torch.float32, requires_grad=True)

  # The sigmoid function gives an output between 0 and 1, which is beneficial when working with classification of binary.
  def f(self, x):
    return torch.sigmoid(self.logits(x))

  def logits(self, x):
      return x @ self.W + self.b

  def loss(self, x, y):
    loss_fn = torch.nn.BCEWithLogitsLoss() # Binary cross entropy with logits.
    return loss_fn(self.logits(x), y)

model = NOTOperatorModel()

# Train the model
optimizer = torch.optim.SGD([model.W, model.b], lr=0.1)
for epoch in range(200000):
  model.loss(x_train, y_train).backward()
  optimizer.step()
  optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W.item(), model.b.item(), model.loss(x_train, y_train).item()))

plt.plot(x_train, y_train, 'o', label='Observations')
plt.xlabel('x')
plt.ylabel('y')
x = torch.arange(0.0, 1.0, 0.01, dtype=torch.float32).reshape(-1,1)
plt.plot(x, model.f(x).detach(), label='f(x) = sigmoid(xW + b)')
plt.legend()
plt.show()