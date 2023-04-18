#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define the number of neurons in the hidden layer
num_hidden_neurons = 20

# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(8, num_hidden_neurons)
        self.out = nn.Linear(num_hidden_neurons, 2)

    def forward(self, inputs):#still problematic with torch.cat
        x = torch.cat((inputs[0], inputs[1]), dim=1)
        x = torch.cat((x, inputs[2]), dim=1)
        x = torch.cat((x, inputs[3]), dim=1)
        x = torch.cat((x, self.prev_outputs), dim=1)
        x = torch.cat((x, self.prev_inputs), dim=1)
        x = torch.sigmoid(self.hidden(x))
        x = self.output(x)
        return x

# Define the loss function
loss_fn = nn.MSELoss()

# Define the optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

# Generate random data for example
data = np.random.rand(10000, 6)
data[:, 4] = np.sin(np.pi * data[:, 0])  # h(t-1)
data[:, 5] = np.sin(np.pi * data[:, 1])  # w(t-1)
data[:, 0] = np.sin(np.pi * data[:, 0])  # v_t
data[:, 1] = np.sin(np.pi * data[:, 1])  # v_w

# Split the data into training, validation, and test sets
train_data, test_data = train_test_split(data, test_size=0.3)
test_data, val_data = train_test_split(test_data, test_size=0.5)

# Convert the data into PyTorch tensors
train_data = torch.from_numpy(train_data).float()
test_data = torch.from_numpy(test_data).float()
val_data = torch.from_numpy(val_data).float()

# Train the model
num_epochs = 1000
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    # Forward pass
    train_outputs = mlp(train_data[:, 0:8])
    train_loss = loss_fn(train_outputs, train_data[:, 8:10])

    # Backward pass and optimization
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Calculate validation loss
    with torch.no_grad():
        val_outputs = mlp(val_data[:, 0:8])
        val_loss = loss_fn(val_outputs, val_data[:, 8:10])

    # Print loss for this epoch
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    print("Epoch [{}/{}], Train Loss: {:.4f}, Validation Loss: {:.4f}"
          .format(epoch+1, num_epochs, train_loss.item(), val_loss.item()))

# Evaluate the model on the test set
with torch.no_grad():
    test_outputs = mlp(test_data[:, 0:8])
    test_loss = loss_fn(test_outputs, test_data[:, 8:10])
    print("Test Loss: {:.4f}".format(test_loss.item()))

# Plot the training and validation loss curves
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:




