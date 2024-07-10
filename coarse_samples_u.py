from problem import *
from coarse_data_generation import *

import torch
import numpy as np
import matplotlib.pyplot as plt
from fem import *
import pandas as pd
import timeit

torch.set_default_dtype(torch.float32)


#Change samples_u to sample from the FEM solution on more than just the coarse grid.


# TO DO 5/4:
#Then in my fine training, I want to minimize the mismatch |FE(x,y)-NN(x,y)|, where
# (x,y) are sampled from a coarse grid


samples_u = torch.tensor(data_points[:,2])#u_true(torch.tensor(coarse_xy[:,0]),torch.tensor(coarse_xy[:,1]))

# print("samples u version", samples_u._version)

samples_u.requires_grad=True
#Add noise to data
mean = 0
std = 1
samples_u = samples_u#+torch.normal(mean,std,samples_u.shape)

# print("samples_u is ",samples_u)

plt.scatter(coarse_xy[:,0],coarse_xy[:,1])
plt.title("Coarse Data Samples")
# plt.show()

print("Samples_u is lookin' good ")