#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Define the ANFIS model
class ANFIS(nn.Module):
    def __init__(self, num_inputs, num_rules, num_outputs):
        super(ANFIS, self).__init__()
        self.num_inputs = num_inputs
        self.num_rules = num_rules
        self.num_outputs = num_outputs
        self.fuzzifier = nn.Parameter(torch.randn(num_rules, num_inputs))
        self.consequent = nn.Parameter(torch.randn(num_rules, num_outputs))
        self.sigma = nn.Parameter(torch.randn(num_outputs))

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.num_rules, 1)
        fuzzy_sets = torch.exp(-torch.sum((x - self.fuzzifier)**2, dim=2) / self.sigma**2)
        normalized_fuzzy_sets = fuzzy_sets / torch.sum(fuzzy_sets, dim=1, keepdim=True)
        output = torch.sum(normalized_fuzzy_sets.unsqueeze(2) * self.consequent.unsqueeze(0), dim=1)
        return output

    
    
# Define the training data (to be substitued by a read.csv)
x = torch.randn(1000, 4)
y = torch.randn(1000, 3)


# Define the ANFIS model
model = ANFIS(num_inputs=4, num_rules=5, num_outputs=3)



# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the ANFIS model
batch_size = 32
num_epochs = 100
train_dataset = TensorDataset(x, y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the ANFIS model
test_x = torch.randn(10, 4)
test_y = torch.randn(10, 3)
with torch.no_grad():
    test_outputs = model(test_x)
    test_loss = criterion(test_outputs, test_y)
print(f'Test Loss: {test_loss.item():.4f}')

