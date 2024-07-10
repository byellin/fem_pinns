from problem import *
from interior_data_generation import *
from coarse_boundary_generation import *

coarse_training_data = torch.cat((coarse_interior,coarse_boundary_points),dim=0)

# print("coarse_train_data type", type(coarse_training_data))

torch.set_default_dtype(torch.float32)
samples_f = torch.tensor(f_true(coarse_training_data[:,0], coarse_training_data[:,1]))
mean = 0
std = 10**-2
samples_f = samples_f#+torch.normal(mean,std,samples_f.shape)
# print("inside samples_f")

# print("coarse boundary points shape", coarse_boundary_points.shape)
# print("coarse training data.shape",coarse_training_data.shape)
# print('samples_f shape is',samples_f.shape)
# print("samples_f is lookin good")