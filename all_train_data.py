from coarse_boundary_generation import *
from interior_data_generation import *

# print(coarse_interior)
# print(coarse_boundary_points)



train_data = torch.cat((coarse_interior,coarse_boundary_points),dim=0)

# print(train_data.shape)

# print(train_data.dtype)

train_data=train_data.to(torch.float32)

# plt.scatter(train_data[:,0], train_data[:,1])
# plt.title("ALL TRAIN DATA FILE. ARE WE ALL ON THE BOUNDARY HERE?")
# plt.show()