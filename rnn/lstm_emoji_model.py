import numpy as np
import torch
import torch.nn as nn

# Initialize training data.

# Dictionary ("hash table") with description and Unicode characters for the emojis.
emojis = {
  'hat': '\U0001F3A9',
  'rat': '\U0001F400',
  'cat': '\U0001F408',
  'flat': '\U0001F3E2',
  'matt': '\U0001F468',
  'cap': '\U0001F9E2',
  'son': '\U0001F466'
}
emoji_encoding = np.eye(len(emojis), dtype=np.float32)
emoji_encoding_size = len(emoji_encoding)

index_to_char = [' ', 'h', 'a', 't', 'r', 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n']
char_encodings = np.eye(len(index_to_char), dtype=np.float32)
char_encoding_size = len(char_encodings)

# Input sequence:
x_train = torch.tensor([
  [[char_encodings[1]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]], # "hat "
  [[char_encodings[4]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]], # "rat "
  [[char_encodings[5]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]], # "cat "
  [[char_encodings[6]], [char_encodings[7]], [char_encodings[2]], [char_encodings[3]]], # "flat"
  [[char_encodings[8]], [char_encodings[2]], [char_encodings[3]], [char_encodings[3]]], # "matt"
  [[char_encodings[5]], [char_encodings[2]], [char_encodings[9]], [char_encodings[0]]], # "cap "
  [[char_encodings[10]], [char_encodings[11]], [char_encodings[12]], [char_encodings[0]]] # "son "
], dtype=torch.float32)

# Output sequence.
y_train = torch.tensor([
  [emoji_encoding[0], emoji_encoding[0], emoji_encoding[0], emoji_encoding[0]], # emoji for a hat
  [emoji_encoding[1], emoji_encoding[1], emoji_encoding[1], emoji_encoding[1]], # emoji for a rat
  [emoji_encoding[2], emoji_encoding[2], emoji_encoding[2], emoji_encoding[2]], # emoji for a cat
  [emoji_encoding[3], emoji_encoding[3], emoji_encoding[3], emoji_encoding[3]], # emoji for a flat
  [emoji_encoding[4], emoji_encoding[4], emoji_encoding[4], emoji_encoding[4]], # emoji for a matt
  [emoji_encoding[5], emoji_encoding[5], emoji_encoding[5], emoji_encoding[5]], # emoji for a cap
  [emoji_encoding[6], emoji_encoding[6], emoji_encoding[6], emoji_encoding[6]]  # emoji for a son
], dtype=torch.float32)


# The purpose of the LSTM model is to learn the corresponding emoji for the word. Many to one.
class LongShortTermMemoryModel(nn.Module):

  def __init__(self, unique_char, unique_emoji):
    super(LongShortTermMemoryModel, self).__init__()
    self.cell_state = None
    self.hidden_state = None

    self.lstm = nn.LSTM(unique_char, 128)  # 128 is the state size, the memory capacity of the LSTM.
    self.dense = nn.Linear(128, unique_emoji) # Convert the 128-sized input to the number of unique emojis.

  def reset(self):  # Reset states prior to a new input sequence.
    zero_state = torch.zeros(1, 1, 128)  # Shape: (number of layers, batch size, state size)
    self.hidden_state = zero_state # LSTM needs a hidden state to remember information across time steps.
    self.cell_state = zero_state # LSTM needs a cell state to remember information across time steps.

  # Processes the input sequence x using LSTM.
  def logits(self, x):
    out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
    # Pass the output through the dense layer to predict the next character in the sequence.
    return self.dense(out.reshape(-1, 128))

  # Prediction function: Utilizes softmax to convert the logits into probabilities.
  def f(self, x):
    return torch.softmax(self.logits(x), dim=1) # Returns a tensor with the probabilist for each class.

  # Calculates the error between the predicted sequence x and the actual sequence y.
  def loss(self, x, y):
    return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

model = LongShortTermMemoryModel(char_encoding_size, emoji_encoding_size)

optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
for epoch in range(500): # Outer loop for multiple passes (epochs) over the training dataset.
  for i in range(x_train.size()[0]): # Inner loop to process each training sample one by one.
    model.reset()
    model.loss(x_train[i], y_train[i]).backward()
    optimizer.step()
    optimizer.zero_grad()

def index_to_emoji(index):
  emoji_list = list(emojis.values())
  return emoji_list[index]

def get_emoji(description):
  model_predictions = []
  model.reset()  # Rest model to prevent previous predictions to affect the next one.
  for k in range(len(description)):
    char_index = index_to_char.index(description[k]) # Finds the index of the char.
    char_vector = char_encodings[char_index] # Retrieves the one-hot encoded vector for the character.
    model_input = torch.tensor([[char_vector]], dtype=torch.float32)
    model_predictions = model.f(model_input) # The model returns a tensor of softmax probabilities for each emoji class.

  highest_probability = model_predictions.argmax(1) # The index of the highest probability in the predictions.
  print(index_to_emoji(highest_probability))

get_emoji("rt")
get_emoji("rats")
get_emoji("sn")



