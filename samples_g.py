from interior_data_generation import *
from problem import *

m=500

torch.set_default_dtype(torch.float32)

#Left boundary, x=-1 
left_samples = domain_y[0]+(domain_y[1]-domain_y[0])*torch.linspace(0,1,m)
samples_g_left = torch.zeros_like(left_samples) #+np.random.normal(mu,sigma,left_samples.shape[0])
samples_g_left
# print("bippaty boppaty boo")
# print("samples_g_left shape", samples_g_left.shape)
# print("samples_g_left", samples_g_left)

#Right boundary, x=1
right_samples = domain_y[0]+(domain_y[1]-domain_y[0])*torch.linspace(0,1,m)
samples_g_right = torch.ones_like(right_samples)#torch.tensor(g_true(1*torch.ones(m), right_samples)) #+np.random.normal(mu,sigma,right_samples.shape[0])
samples_g_right

#Top boundary, y=1
top_samples = domain_y[0]+(domain_y[1]-domain_y[0])*torch.linspace(0,1,m)
samples_g_top = torch.ones_like(top_samples)#torch.tensor(g_true(top_samples, 1*torch.ones(m))) #+np.random.normal(mu,sigma,top_samples.shape[0])
samples_g_top

#Bottom boundary, y=-1
bottom_samples = domain_y[0]+(domain_y[1]-domain_y[0])*torch.linspace(0,1,m)
samples_g_bottom = torch.ones_like(bottom_samples)#torch.tensor(g_true(bottom_samples, -1*torch.ones(m))) #+np.random.normal(mu,sigma,bottom_samples.shape[0])
samples_g_bottom


coarse_xy_top = torch.cat((top_samples.reshape(-1, 1), domain_y[1] * torch.ones(m, 1)), dim=1)
coarse_xy_bottom = torch.cat((bottom_samples.reshape(-1, 1), domain_y[0] * torch.ones(m, 1)), dim=1)
coarse_xy_left = torch.cat(( domain_x[0] * torch.ones(m, 1),left_samples.reshape(-1, 1)), dim=1)
coarse_xy_right = torch.cat(( domain_x[1] * torch.ones(m, 1),right_samples.reshape(-1, 1)), dim=1)
coarse_xy_boundary = torch.cat((coarse_xy_top,coarse_xy_bottom,coarse_xy_left,coarse_xy_right),dim=0)

# print("samples_g_left is type",type(samples_g_left))
# print(samples_g_left)


# print("samples_g_right is type",type(samples_g_right))
# print(samples_g_right)
#
# print("g_true(-1*torch.ones(m), left_samples)", g_true(-1*torch.ones(m), left_samples))

samples_g_train = torch.cat((samples_g_left,samples_g_right,samples_g_top,samples_g_bottom),dim=0)

print("Boundary data generation looking good")
