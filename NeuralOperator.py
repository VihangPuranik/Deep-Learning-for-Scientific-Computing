#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
from NerualOperatorClass import FeedForwardNN, DeepOnet
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# In[3]:


n_train = 300
n_input_sensor = 200
n_output_sensor = 250


x_data = torch.from_numpy(np.load("data/AC_data_input.npy")).type(torch.float32)
y_data = torch.from_numpy(np.load("data/AC_data_output.npy")).type(torch.float32)


input_sensor_idx = np.random.randint(0, x_data.shape[1], n_input_sensor)
output_sensor_idx = np.random.randint(0, x_data.shape[1], n_output_sensor)

input_function_train = x_data[:n_train,  input_sensor_idx, 1]
output_function_train = y_data[:n_train, output_sensor_idx]
input_function_test = x_data[n_train:, input_sensor_idx, 1]
output_function_test = y_data[n_train:, output_sensor_idx]

sensor_inputs = x_data[0, input_sensor_idx, 0].unsqueeze(1)
y = x_data[0, output_sensor_idx, 0].unsqueeze(1)

print(input_function_train.shape, y.shape)
print(output_function_train.shape)


fig, axes = plt.subplots(2, 2, figsize=(30, 8), dpi=200)
axes[0,0].grid(True, which="both", ls=":")
axes[0,1].grid(True, which="both", ls=":")
axes[1,0].grid(True, which="both", ls=":")
axes[1,1].grid(True, which="both", ls=":")

axes[0,0].scatter(sensor_inputs, input_function_train[0], label=r'$u_0(x)$')
axes[0,0].scatter(y, output_function_train[0], label=r'$u_T(x)$')

axes[0,1].scatter(sensor_inputs, input_function_train[1], label=r'$u_0(x)$')
axes[0,1].scatter(y, output_function_train[1], label=r'$u_T(x)$')

axes[1,0].scatter(sensor_inputs, input_function_train[2], label=r'$u_0(x)$')
axes[1,0].scatter(y, output_function_train[2], label=r'$u_T(x)$')

axes[1,1].scatter(sensor_inputs, input_function_train[3], label=r'$u_0(x)$')
axes[1,1].scatter(y, output_function_train[3], label=r'$u_T(x)$')
axes[0, 0].set( title="Sample 1")
axes[0, 1].set( title="Sample 2")
axes[1, 0].set(xlabel=r'$x$',title="Sample 3")
axes[1, 1].set(xlabel=r'$x$', title="Sample 4")
axes[0, 0].legend()
axes[0, 1].legend()
axes[1, 0].legend()
axes[1, 1].legend()
#axes[1].set(xlabel=r'$x$', ylabel=r'$y$', title="Exact Coefficient " + r'$a(x,y)$')
#axes[2].set(xlabel=r'$x$', ylabel=r'$y$', title="Predicted Coefficient " + r'$a^\ast(x,y)$')


# In[ ]:


branch_architecture_ = {
    "n_hidden_layers": 4,
    "neurons": 20,
    "act_string": "tanh",
    "retrain": 65
}

trunk_architecture_ = {
    "n_hidden_layers": 4,
    "neurons": 24,
    "act_string": "tanh",
    "retrain": 125
}
n_basis_function = 10

branch = FeedForwardNN(n_input_sensor, n_basis_function, branch_architecture_)
trunk = FeedForwardNN(1, n_basis_function, trunk_architecture_)
#print(branch(input_function_train))
print(trunk(y))
deeponet = DeepOnet(branch, trunk)
print(input_function_train.shape,y.shape)


# In[ ]:


print(deeponet(input_function_train, y))


# In[ ]:


optimizer = optim.Adam(deeponet.parameters(), lr=0.001, weight_decay=1e-6)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.8)


# In[ ]:



training_set = DataLoader(TensorDataset(input_function_train, output_function_train), batch_size=50, shuffle=True)
l = torch.nn.MSELoss()
epochs = 1000
freq_print = 10


# In[ ]:


for epoch in range(epochs):
    print(epoch)
    if epoch % freq_print == 0: print("################################ ", epoch, " ################################")
    train_mse = 0.0
    for step, (input_batch, output_batch) in enumerate(training_set):

        optimizer.zero_grad()
        #output_pred_batch = deeponet(input_batch, y)
        #print(output_batch.shape, output_pred_batch.shape)
        #loss_f = l(output_pred_batch, output_batch)
        #loss_f.backward()
        loss_f=torch.tensor(0.)
        optimizer.step()
        train_mse += loss_f.item()
    train_mse /= len(training_set)
    scheduler.step()


# In[ ]:




