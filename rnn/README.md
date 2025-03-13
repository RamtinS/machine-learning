# Recurrent Neural Networks (RNN)  

## What is a Recurrent Neural Network (RNN)?  

Recurrent Neural Networks (RNNs) are a type of neural network designed for sequential data. They maintain memory of previous inputs, making them useful for tasks like text generation and sequence prediction. Long Short-Term Memory (LSTM) networks are a special type of RNN that help handle long-term dependencies.  

## Task Description  

This project involves training LSTM models for character and word-based sequence prediction.  

### Subtasks:  

1. **Many-to-Many LSTM (Character Generation)**  
   - Train an LSTM model on the characters in `"hello world"`.  
   - Use the trained model to generate 50 characters after the input `"h"`.  

2. **Many-to-One LSTM (Word-to-Emoji Mapping)**  
   - Train an LSTM model on words like `"hat"`, `"rat"`, `"cat"`, `"flat"`, `"matt"`, `"cap"`, and `"son"`, with corresponding emojis.  
   - Use character-based encoding and pad shorter words with spaces for batch training.  
   - Test the model on words like `"rt"` and `"rats"` to observe the predicted emoji.  
