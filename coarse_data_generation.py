import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import timeit

torch.set_default_dtype(torch.float32)

#Define domain 
domain_x = [-1,1]
domain_y = [-1,1]


coarse_samples_x, coarse_samples_y = np.mgrid[domain_x[0]:domain_x[1]:10j, domain_y[0]:domain_y[1]:10j]
coarse_xy = np.vstack((coarse_samples_x.flatten(), coarse_samples_y.flatten())).T

coarse_xy = torch.tensor(coarse_xy, dtype=torch.float32)

print("Coarse_data_generation lookin' good ")