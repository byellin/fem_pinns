from boundary_check import *

import torch

import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

# Set the seed for reproducibility
torch.manual_seed(42)

# Generate random numbers for y values
random_numbers = torch.rand(200) * 2 - 1

# Create the set of points for the right side
right_points = torch.stack([torch.ones(200), random_numbers], dim=1)

# Create the set of points for the top side
top_points = torch.stack([random_numbers, torch.ones(200)], dim=1)

# Generate the remaining points for the left and bottom sides
left_points = torch.stack([-torch.ones(200), random_numbers], dim=1)
bottom_points = torch.stack([random_numbers, -torch.ones(200)], dim=1)

# Concatenate all the points
coarse_boundary_points = torch.cat((right_points, top_points, left_points, bottom_points), dim=0)

# Print the points
plt.scatter(coarse_boundary_points[:,0],coarse_boundary_points[:,1])

# boundary_points = np.empty((0,2))
# for i in coarse_xy:
#   x,y = i[0],i[1]
#   #print(i)
#   if on_boundary(x,y):# or on_neumann_boundary(x,y):
#     boundary_points=np.vstack([boundary_points,[x,y]])
#     # print(i)
#   elif on_neumann_boundary(x,y):
#     boundary_points=np.vstack([boundary_points,[x,y]])
    

# plt.scatter([np.squeeze(boundary_points[:,0])],[np.squeeze(boundary_points[:,1])])
# plt.title("Boundary of Domain")
# plt.show()

#This boundary will eventually be replaced in the coarse training with a lot more points,
#but maybe I should just use the coarse boundary in the coarse training and just 
#accept that the results won't be good