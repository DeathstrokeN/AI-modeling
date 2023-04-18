#!/usr/bin/env python
# coding: utf-8

# In[34]:


import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the second-degree polynomial function to fit the data
def second_degree(x, y, a, b, c, d, e, f):
    return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

# Generate random data of length 10000
x_data = 0.5*torch.randn(1000)
y_data = torch.randn(1000)
z_data = 2*x_data**2 + 3*y_data**2 + 4*x_data*y_data + 5*x_data + 6*y_data + 7

# Fit the data using a linear regression
X = torch.stack([x_data**2, y_data**2, x_data*y_data, x_data, y_data, torch.ones_like(x_data)], dim=1)
model = torch.nn.Linear(X.shape[1], 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.1)
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    y_pred = model(X).squeeze()
    loss = criterion(y_pred, z_data)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Compute the second-degree polynomial coefficients from the linear regression coefficients
a, b, c, d, e, f = model.weight.squeeze().tolist()
#d, e, _ = (2*a, 2*b, c)
coefficients = [a, b, c, d, e, f]

# Compute the second-degree polynomial over a grid
xx, yy = torch.meshgrid(torch.linspace(x_data.min(), x_data.max(), 100),
                        torch.linspace(y_data.min(), y_data.max(), 100))
fit = second_degree(xx, yy, *coefficients)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the sample data
ax.scatter(x_data.numpy(), y_data.numpy(), z_data.numpy(), c='r', marker='o')

# Plot the regression surface
ax.plot_surface(xx.numpy(), yy.numpy(), fit.numpy(), color='b', alpha=.5)

# Set the axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()


# In[18]:


x_data


# In[35]:


coefficients


# In[43]:


loss.

