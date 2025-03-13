import numpy as np
import torch
import torch.nn as nn

# Training data.
index_to_char = [' ', 'h', 'e', 'l', 'o', 'w', 'r', 'd'] # The characters the model will work with.

# Creates a square identity matrix with the length of the array as number of rows.
char_encodings = np.eye(len(index_to_char), dtype=np.float32)

encoding_size = len(char_encodings)

# Input sequence: " hello world"
x_train = torch.tensor([[char_encodings[0]], [char_encodings[1]], [char_encodings[2]], [char_encodings[3]],
                        [char_encodings[3]], [char_encodings[4]], [char_encodings[0]], [char_encodings[5]],
                        [char_encodings[4]], [char_encodings[6]], [char_encodings[3]], [char_encodings[7]]],
                        dtype=torch.float32)

# Output sequence: "hello world "
# The output sequence is what the model should learn to predict based on the input sequence x_train.
y_train = torch.tensor([char_encodings[1], char_encodings[2], char_encodings[3], char_encodings[3],
                        char_encodings[4], char_encodings[0], char_encodings[5], char_encodings[4],
                        char_encodings[6], char_encodings[3], char_encodings[7], char_encodings[0]],
                        dtype=torch.float32)

# The purpose of the LSTM model is to learn sequences of characters. Many-to-many model.
class LongShortTermMemoryModel(nn.Module):

  def __init__(self, unique_char): # The number of unique characters.
    super(LongShortTermMemoryModel, self).__init__()
    self.cell_state = None
    self.hidden_state = None

    self.lstm = nn.LSTM(unique_char, 128)  # 128 is the state size, the memory capacity of the LSTM.

    # The dense layer is added to convert the 128-sized output of the LSTM to the encoding size.
    self.dense = nn.Linear(128, unique_char)

  def reset(self):  # Reset states prior to a new input sequence.
    zero_state = torch.zeros(1, 1, 128)  # Shape: (number of layers, batch size, state size)
    self.hidden_state = zero_state # LSTM needs a hidden state to remember information across time steps.
    self.cell_state = zero_state # LSTM needs a cell state to remember information across time steps.

  # Processes the input sequence x using LSTM.
  def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
    out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
    # Pass the output through the dense layer to predict the next character in the sequence.
    return self.dense(out.reshape(-1, 128))

  # Prediction function: Utilizes softmax to convert the logits into probabilities.
  def f(self, x):  # x shape: (sequence length, batch size, encoding size)
    return torch.softmax(self.logits(x), dim=1)

  # Calculates the error between the predicted sequence x and the actual sequence y.
  def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
    return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

model = LongShortTermMemoryModel(encoding_size)

optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
for epoch in range(500):
    model.reset() # Reset the hidden and cell states of the LSTM before each training sample.
    model.loss(x_train, y_train).backward() # Compute the loss and perform backpropagation.
    optimizer.step() # Update the model's weights based on the computed gradients.
    optimizer.zero_grad() # Clear the gradients for the next iteration.

    if epoch % 10 == 9:
        # Generate characters from the initial characters 'h'
        model.reset()
        text = ' h'
        model.f(torch.tensor([[char_encodings[0]]]))
        y = model.f(torch.tensor([[char_encodings[1]]]))
        text += index_to_char[y.argmax(1)]
        for c in range(50):
            y = model.f(torch.tensor([[char_encodings[y.argmax(1)]]]))
            text += index_to_char[y.argmax(1)]
        print(text)
