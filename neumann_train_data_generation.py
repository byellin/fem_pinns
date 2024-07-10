# print("about to import boundary check")
from boundary_check import *
# print("about to import coarse samples u")
print("shoopa doop")
from coarse_boundary_generation import *

from coarse_samples_u import *
# print("shoopa doop 2")



# print("just imported coarse samples u")
# print("about to import coarse data gen")
# from coarse_data_generation import *
# print("About to import packages")
import torch
# print("just imported torch")
import numpy as np
# print("just imported np")
import matplotlib.pyplot as plt
# print("just imported plt")
import pandas as pd
# print("Just imported pandas")
import timeit
# print("just imported timeit")


k=10

torch.set_default_dtype(torch.float32)


neumann_boundary=[]
# dirichlet_boundary = np.empty((0,2))
# print("just after defining empty dirichlet boudary array")
for i in range(coarse_boundary_points.shape[0]):
	if on_neumann_boundary(coarse_boundary_points[i][0],coarse_boundary_points[i][1]):
		neumann_boundary.append((coarse_boundary_points[i][0],coarse_boundary_points[i][1]))
		# dirichlet_boundary[dirichlet_boundary.shape[0],:] = [coarse_xy[i][0],coarse_xy[i][1]]



neumann_train_data_boundary = torch.tensor(neumann_boundary)
# print("neumann bdd dtype is", neumann_boundary.dtype)
# print(neumann_train_data_boundary)
plt.clf()

# plt.scatter(neumann_train_data_boundary[:,0],neumann_train_data_boundary[:,1])
# plt.title("Neumann Train Data Boundary")
# plt.show()
