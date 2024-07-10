import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

#Set data type of values in neural network
torch.set_default_dtype(torch.float64)

#Set the parameter k that appears in the heat equation
k=3

#Set the number of points we define initial condition on
n = 100

#Set the number of time steps
m=20

#Set the initial time
t0 = 0

#Define the x domain
domain_x = torch.linspace(0,1,n)

#Define the t domain
domain_t = torch.linspace(0,1,m)

#Define the initial conditions
u_init = torch.sin(torch.pi*domain_x)

#Plot the initial conditions
plt.scatter(range(len(u_init)),u_init)
plt.title("Initial Data")
plt.show()


print("initial_conditions lookin' good so far")
print("______________________________________")