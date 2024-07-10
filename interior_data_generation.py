import numpy as np
import torch



# print("about to import boundary check")
from boundary_check import *
# print("about to import coarse samples u")
# print("shoopa doop")


from coarse_samples_u import *

from coarse_data_generation import *

import torch

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import timeit

interior=[]

torch.set_default_dtype(torch.float32)

for i in range(coarse_xy.shape[0]):
	if not on_dirichlet_boundary(coarse_xy[i][0],coarse_xy[i][1]) and not on_neumann_boundary(coarse_xy[i][0],coarse_xy[i][1]):
		interior.append((coarse_xy[i][0],coarse_xy[i][1]))
		# dirichlet_boundary[dirichlet_boundary.shape[0],:] = [coarse_xy[i][0],coarse_xy[i][1]]

coarse_interior = torch.tensor(interior)
# print(coarse_interior)
# print("inside of interior data generation")
# print("The shape of coarse_interior is ",coarse_interior.shape)

# print(dirichlet_boundary)
# print("All lookin good")