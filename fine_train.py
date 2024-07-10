# train neural network

import numpy as np
import torch
import pandas as pd 
from nn_fine import *
from dirichlet_train_data_generation import *
from dirichlet_boundary_loss_func import * 
from neumann_boundary_loss_func import *
from neumann_train_data_generation import *
from interior_loss_func import *
from all_train_data import *
from samples_f import *
from samples_g import *
from quadrature import *
from weights import *


#Plot the results of the coarse training on the fine data

#Plot the results from the coarse training on fine scale data

#5/8/23

#Liz said what's probably happening is that this example is too simple. So the results
#from the coarse training are already good enough for the fine training. And then
#when the fine training starts, the network doesn't really need to update parameters by much.
#These plots show how the coarse trained model does on fine scale data before any secondary fine training.

#I should try this on a harder problem, like u(x,y) = sin(x)cos(y) and see if the model is improved in the
#fine scale training.

#I DON'T THINK MY INITIAL GUESS IS SHOWING CORRECTLY. BECAUSE THE PLOT ALL THE WAY ON THE LEFT IS JUST CONSTANTLY 0


domain_x = [-1,1]
domain_y = [-1,1]

train_data_boundary_x = pd.read_csv("train_data_boundary_x.csv")
train_data_boundary_x = torch.tensor(train_data_boundary_x.iloc[:,1])


train_data_boundary_y = pd.read_csv("train_data_boundary_y.csv")
train_data_boundary_y = torch.tensor(train_data_boundary_y.iloc[:,1])

print("BEFORE RESHAPING train_data_boundary_x and y ")
print("train_data_boundary_x.shape is ", train_data_boundary_x.shape)
print("train_data_boundary_y.shape is ", train_data_boundary_y.shape)
print("Train data boundary x is ", train_data_boundary_x)

fine_samples_x, fine_samples_y = np.mgrid[domain_x[0]:domain_x[1]:100j, domain_y[0]:domain_y[1]:100j]

fine_samples_x = torch.tensor(fine_samples_x)

fine_samples_y = torch.tensor(fine_samples_y)
fine_xy = np.vstack((fine_samples_x.flatten(), fine_samples_y.flatten())).T

fine_samples_x = torch.tensor(fine_xy[:,0])
fine_samples_y = torch.tensor(fine_xy[:,1])


print("JUST LOOK HERE: The shape of fine_xy is", fine_xy.shape)

fine_samples_x.requires_grad = True
fine_samples_y.requires_grad = True
 
print("fine_samples_x shape is", torch.tensor(fine_samples_x).shape)
print("train_data_boundary_x shape is ", torch.tensor(train_data_boundary_x).unsqueeze(1).shape)

fine_train_data_x_with_bdd = torch.cat((torch.tensor(fine_samples_x.unsqueeze(1)), torch.tensor(train_data_boundary_x).unsqueeze(1)), axis=0)

fine_train_data_y_with_bdd = torch.cat((torch.tensor(fine_samples_y.unsqueeze(1)), torch.tensor(train_data_boundary_y).unsqueeze(1)), axis=0)




#Still need to remove the boundary points from fine_xy
fine_xy = np.hstack((fine_samples_x.detach().reshape(-1,1),fine_samples_y.detach().reshape(-1,1)))





#Set parameters from coarse training and check that they got transferred correctly

model_fine = NeuralNetwork_fine()


torch.set_default_dtype(torch.float32)

for param in model_fine.linear_relu_stack.parameters():
  print(param.data.dtype)

checkpt = torch.load("coarse_result.pth", map_location=lambda storage, loc: storage)
model_fine.load_state_dict(checkpt["state_dict"])

print("Here are the parameters that we are using from the coarse training to initialize the second network")
print("ROUND 1 Parameters")
for parameter in model_fine.parameters():
  print(parameter)
  print(parameter.shape)

import matplotlib.pyplot as plt
# fig = plt.gcf()
import matplotlib as mpl
norm = mpl.colors.Normalize(vmin=0, vmax=6)
plt.figure(figsize = (15,5))

samples_u = u_true(fine_train_data_x_with_bdd,fine_train_data_y_with_bdd)
#THIS IS WHERE I SHOULD PULL IN THE DATA FROM THE FEM INITIALIZATION


# torch.set_default_dtype(torch.float32)
fine_train_data_x_with_bdd= torch.tensor(fine_train_data_x_with_bdd,dtype=torch.float32)
fine_train_data_y_with_bdd= torch.tensor(fine_train_data_y_with_bdd,dtype=torch.float32)

Z = model_fine(torch.cat((fine_train_data_x_with_bdd,fine_train_data_y_with_bdd),dim=1))


plt.subplot(1,3,1)
plt.scatter(fine_train_data_x_with_bdd.detach().numpy(),fine_train_data_y_with_bdd.detach().numpy(),c=Z.detach().numpy())
# plt.clim(-1,1)
plt.colorbar(norm=norm)
# plt.gray()
plt.title("Coarse Trained Model Prediction on Fine Data")


plt.subplot(1,3,2)
plt.scatter(fine_train_data_x_with_bdd.detach().numpy(),fine_train_data_y_with_bdd.detach().numpy(),c=samples_u.detach().numpy())
# plt.clim(-1,1)
plt.colorbar(norm=norm)
# plt.gray()
plt.title("True solution on Training data")



plt.subplot(1,3,3)
plt.scatter(fine_train_data_x_with_bdd,fine_train_data_y_with_bdd,c=(Z.detach().numpy()-samples_u.detach().numpy()))
# plt.clim(-2.6,0.2)
plt.colorbar(norm=norm)
# plt.gray()
plt.title("Difference between model and true solution")

plt.show()


optimizer = torch.optim.LBFGS(model_fine.parameters(),max_iter=70,line_search_fn='strong_wolfe')

#Initialize list of losses - keep track of interior and boundary losses separately
losses=[]
boundary_losses = []
interior_losses = []
neumann_bdd_losses = []
dirichlet_bdd_losses = []
data_fit_losses = []
physics_losses = []


#Change this back to 250 once I have the error calculation working
num_epochs = 100

def closure_fine():
  torch.set_default_dtype(torch.float32)
  optimizer.zero_grad()


  # data_fit_interior_weight = 7
  # data_fit_neumann_bdd_weight = 5
  # data_fit_dirichlet_bdd_weight = 100
  # data_fit_interior_weight = 10
  # interior_loss_func_weight = 0#3

  #Need to add data fitting term to my obejctive function (DO THIS IN THE AFTERNOON 3/22)

  #Call fine trained model 
  output_full_interior = model_fine(torch.tensor(coarse_xy))
  # print(output_full_interior.shape)
  # output_full_neumann_boundary = model_fine(torch.tensor(neumann_boundary))
  # print(output_full_neumann_boundary.shape)
  output_full_dirichlet_boundary = model_fine(torch.tensor(dirichlet_boundary))
  # print(output_full_dirichlet_boundary.shape)

  #Interior data fit term
  data_fit_interior_loss=torch.norm(output_full_interior-torch.reshape(u_true(torch.tensor(coarse_xy[:,0]),torch.tensor(coarse_xy[:,1])),(-1,1)))

  #CHANGE THIS DEIFNITION TO MAKE SURE MODEL FITS THE DATA ON THE BOUNDARY

  #I think these are redundant
  # data_fit_loss_neumann_bdd = data_fit_neumann_bdd_weight*torch.norm(output_full_neumann_boundary-torch.reshape(u_true(neumann_train_data_boundary[:,0],neumann_train_data_boundary[:,1]),(-1,1)))
  # data_fit_loss_dirichlet_bdd = data_fit_dirichlet_bdd_weight*torch.norm(output_full_dirichlet_boundary-torch.reshape(u_true(dirichlet_train_data_boundary[:,0],dirichlet_train_data_boundary[:,1]),(-1,1)))

  samples_g_train=g_true(dirichlet_boundary[:,0],dirichlet_boundary[:,1])
  # print("the type of neumann bdd is ", type(neumann_boundary))
  objective = data_fit_dirichlet_bdd_weight*dirichlet_boundary_loss_func(dirichlet_boundary[:,0].unsqueeze(dim=1),
                                 dirichlet_boundary[:,1].unsqueeze(dim=1),
                                 samples_g_train.unsqueeze(dim=1),
                                  model_fine) + interior_loss_func_weight_fine*interior_loss_func(torch.tensor(train_data[:,0]).unsqueeze(dim=1),torch.tensor(train_data[:,1]).unsqueeze(dim=1),samples_f,model)+data_fit_interior_weight_fine*data_fit_interior_loss#+data_fit_neumann_bdd_weight*neumann_boundary_loss_func(torch.tensor(neumann_boundary)[:,0].unsqueeze(dim=1),
                                 # torch.tensor(neumann_boundary)[:,1].unsqueeze(dim=1),model_fine)

  #I need to add a term checking the data fitting on coarse_xy to my objective function. I don't think that's there yet


  # print("The Neumann boundary loss is ", neumann_boundary_loss_func(1,1))
  objective.backward(retain_graph=True)
  return objective

#Train neural network with physics!
torch.set_default_dtype(torch.float32)
import timeit
tic = timeit.default_timer()
for epoch in range(num_epochs):

  # for parameter in model_fine.parameters():
  #   print(parameter)

    # print(parameter.shape)

  parameters = [p for p in model_fine.parameters() if p.grad is not None and p.requires_grad]
  
  total_norm = 0
  
  for p in parameters:
    param_norm = p.grad.detach().data.norm(2)
    total_norm += param_norm.item() ** 2
  total_norm = total_norm ** 0.5
  # print("here's the dirichlet boundary")
  # print(dirichlet_boundary)
  # print("Inside train.py The length of the dirichlet boundary is ",len(dirichlet_boundary))
  n = torch.tensor(dirichlet_boundary).shape[0]#+torch.tensor(neumann_boundary).shape[0]

  #Set batch size
  interior_batch_size = 20

  #boundary_batch_size = int(n/(int(train_data.shape[0]/interior_batch_size)))

  optimizer.step(closure_fine)

  #Randomly order the indices of the boundary data points
  idx_bdd = torch.randperm(n)

  idx_int = torch.randperm(train_data.shape[0])


  #Evaluate the model on the training data
  fine_xy = torch.tensor(fine_xy,dtype=torch.float32)
  output_interior = model_fine(torch.tensor(fine_xy))
  output_dirichlet = model_fine(torch.tensor(dirichlet_boundary))
  # output_neumann = model_fine(torch.tensor(neumann_boundary))

 
  # dirichlet_boundary_weight = 10
  # neumann_boundary_weight = 3
  # interior_loss_weight = 3
  # data_fit_weight = 10

  dirichlet_loss = data_fit_dirichlet_bdd_weight * dirichlet_boundary_loss_func(dirichlet_boundary[:,0].unsqueeze(dim=1),
                                              dirichlet_boundary[:,1].unsqueeze(dim=1),
                                              samples_g_train.unsqueeze(dim=1),model_fine)

  # neumann_loss = neumann_boundary_weight * neumann_boundary_loss_func(torch.tensor(neumann_boundary)[:,0].unsqueeze(dim=1),torch.tensor(neumann_boundary)[:,1].unsqueeze(dim=1),model_fine)


  boundary_loss = data_fit_dirichlet_bdd_weight*dirichlet_loss #+ neumann_boundary_weight*neumann_loss



  # print("Dirichlet loss: ",dirichlet_boundary_loss_func(dirichlet_train_data_boundary[:,0].unsqueeze(dim=1),
  #                                            dirichlet_train_data_boundary[:,1].unsqueeze(dim=1),
  #                                            samples_g_train.unsqueeze(dim=1),model))

  #print("Neumann loss: ", neumann_loss)
  #print(boundary_loss)


  #4/27 I think I should replace what this loss is being calculated on.
  #I think it should be coarse XY instead of train_data

  #interior_loss = interior_loss_func(torch.tensor(coarse_xy[:,0]).unsqueeze(dim=1),torch.tensor(coarse_xy[:,1]).unsqueeze(dim=1),samples_f)

  #losses.append(interior_loss+boundary_loss)
  #losses.append(boundary_loss)

  # print("output full shape ",output_full.shape)
  # print("true soln on train data ",torch.reshape(u_true(train_data[:,0],train_data[:,1]),(-1,1)).shape)


  data_fit_loss_interior = torch.norm(output_interior-torch.reshape(u_true(torch.tensor(fine_xy[:,0]),torch.tensor(fine_xy[:,1])),(-1,1)))
  # data_fit_loss_neumann = torch.norm(output_neumann-torch.reshape(u_true(neumann_train_data_boundary[:,0],neumann_train_data_boundary[:,1]),(-1,1)))
  data_fit_loss_dirichlet = torch.norm(output_dirichlet-torch.reshape(u_true(dirichlet_boundary[:,0],dirichlet_boundary[:,1]),(-1,1)))
  # physics_loss = interior_loss_func(train_data[:,0].unsqueeze(dim=1),train_data[:,1].unsqueeze(dim=1),samples_f)

  # interior_losses.append(data_fit_loss_interior)
  # neumann_bdd_losses.append(data_fit_loss_neumann)
  # dirichlet_bdd_losses.append(data_fit_loss_dirichlet)



  # data_fit_losses.append(data_fit_loss)

  interior_loss = data_fit_loss_interior

  #Also store the interior and boundary losses separately
  interior_losses.append(interior_loss)
  boundary_losses.append(data_fit_loss_dirichlet)#+data_fit_loss_neumann)

  #Split up the boundary losses into Dirichlet and Neumann
  dirichlet_bdd_losses.append(dirichlet_loss)
  # neumann_bdd_losses.append(neumann_loss)

  #Store the physics losses
  # physics_losses.append(physics_loss)
  #print(interior_loss)

  #Append the total loss (interior+boundary)
  #losses.append(boundary_loss)




  #Keepin' you in the loop about how your training is going
  # Keepin' you in the loop about how your training is going
  if epoch % 10 == 0:
      # print("boop")
      print('Epoch' + str(epoch))
      # print('total: ', losses[-1])
      print('Data fit loss interior: ', interior_losses[-1].item())
      # print('Data fit loss boundary: ', boundary_losses[-1].item())
      # print('Data fit: ', torch.norm(output_interior - torch.reshape(u_true(coarse_xy[:, 0], coarse_xy[:, 1]), (-1, 1))))
      print('Data fit loss Dirichlet', dirichlet_bdd_losses[-1].item())
      # print('Data fit loss Neumann', neumann_bdd_losses[-1].item())
      # print('Physics Loss', physics_losses[-1].item())

toc = timeit.default_timer()

print(toc-tic, str(" seconds to train")) #tic-toc on the clock but the party don't stop



#After the second training 

Z2 = model_fine(torch.cat((fine_train_data_x_with_bdd,fine_train_data_y_with_bdd),dim=1))

#Print parameters after the second training 
print("ROUND 2 PARAMETERS")
for param in model_fine.linear_relu_stack.parameters():
  # print(param.data.dtype)
  print(param)

#THIS PLOT ISN'T SHOWING THE RIGHT THING. IT'S SHOWING THE EXACT SAME THING AS THE FIRST PLOT 
#BEFORE THE SECOND TRAINING 
plt.subplot(1,3,1)
plt.scatter(fine_train_data_x_with_bdd.detach().numpy(),fine_train_data_y_with_bdd.detach().numpy(),c=Z2.detach().numpy())
# plt.clim(-1,1)
plt.colorbar(norm=norm)
# plt.gray()
plt.title("Fine Trained Model Prediction on Fine Data")


plt.subplot(1,3,2)
plt.scatter(fine_train_data_x_with_bdd.detach().numpy(),fine_train_data_y_with_bdd.detach().numpy(),c=samples_u.detach().numpy())
# plt.clim(-1,1)
plt.colorbar(norm=norm)
# plt.gray()
plt.title("True solution on Training data")



plt.subplot(1,3,3)
plt.scatter(fine_train_data_x_with_bdd,fine_train_data_y_with_bdd,c=(Z2.detach().numpy()-samples_u.detach().numpy()))
# plt.clim(-2.6,0.2)
plt.colorbar(norm=norm)
# plt.gray()
plt.title("Difference between model and true solution")

print("THIS IS WHERE I AM TRYING TO INTEGRATE MY ERROR")

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt



import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# print("THE SHAPE OF Z2 is ",Z2.shape)
#
# # Assuming you have a 2D domain represented by x and y coordinates
# #My fine training is initially done on a 100x100 grid,
# #so I'm testing my learned function on a 200x200 grid
x = np.linspace(-1, 1, num=200)
y = np.linspace(-1, 1, num=200)

# # Create a grid of points in the 2D domain
# x_grid, y_grid = np.meshgrid(x, y)
#
#
#

#
# # Check data types
# print("the type of fine samples x is ",fine_samples_x_test.dtype)
# print("the type of fine samples y is ", fine_samples_y_test.dtype)
#
# # Convert data types if needed
# if fine_samples_x_test.dtype != fine_samples_y_test.dtype:
#     fine_samples_x_test = fine_samples_x_test.to(fine_samples_y_test.dtype)
#
# print("now the type of tensor fine samples x is ", torch.tensor(fine_samples_x_test).dtype)
#

#Test model on unseen data
# fine_samples_x_test, fine_samples_y_test = np.mgrid[domain_x[0]:domain_x[1]:200j, domain_y[0]:domain_y[1]:200j]
# fine_xy_test = np.hstack((fine_samples_x_test.reshape(-1,1),fine_samples_y_test.reshape(-1,1)))
#
# print("fine samples x dtype is ", fine_samples_x_test.dtype)
#
# Z3 = model_fine(torch.cat((torch.tensor(fine_samples_x_test),torch.tensor(fine_samples_y_test)),dim=1))
# # print("Z3 dtype is ", Z3.dtype)
# # print()
# plt.scatter(fine_samples_x_test.reshape(-1,1),fine_samples_y_test.reshape(-1,1),c=Z3.detach().numpy())
# plt.title("Fine Trained Model Prediction on Test Data")
# plt.show()


fine_samples_x_test, fine_samples_y_test = np.mgrid[domain_x[0]:domain_x[1]:200j, domain_y[0]:domain_y[1]:200j]

fine_samples_x_test = torch.tensor(fine_samples_x)
fine_samples_y_test = torch.tensor(fine_samples_y)



# fine_samples_x_test
fine_xy_test = np.vstack((fine_samples_x_test.flatten(), fine_samples_y_test.flatten())).T

fine_samples_x_test = torch.tensor(fine_xy_test[:, 0], dtype=torch.float32)
fine_samples_y_test = torch.tensor(fine_xy_test[:, 1], dtype=torch.float32)



# print("JUST LOOK HERE: The shape of fine_xy_test is", fine_xy_test.shape)

fine_samples_x_test.requires_grad = True
fine_samples_y_test.requires_grad = True

# print("fine_samples_x shape is", torch.tensor(fine_samples_x).shape)
# print("train_data_boundary_x shape is ", torch.tensor(train_data_boundary_x).unsqueeze(1).shape)

fine_train_data_x_test_with_bdd = torch.cat(
  (torch.tensor(fine_samples_x_test.unsqueeze(1)), torch.tensor(train_data_boundary_x).unsqueeze(1)), axis=0)

fine_train_data_y_test_with_bdd = torch.cat(
  (torch.tensor(fine_samples_y.unsqueeze(1)), torch.tensor(train_data_boundary_y).unsqueeze(1)), axis=0)

# Still need to remove the boundary points from fine_xy
fine_xy_test = np.hstack((fine_samples_x_test.detach().reshape(-1, 1), fine_samples_y_test.detach().reshape(-1, 1)))

# print("Let's print some dtypes, man")
#
# print(fine_samples_y_test.dtype)
# print(fine_samples_x_test.dtype)

x_grid = np.linspace(-1,1,100)
y_grid = np.linspace(-1,1,100)
X, Y = np.meshgrid(x_grid, y_grid)

samples_u = u_true(X,Y)

Z2 = model_fine(torch.cat((torch.tensor(X,dtype=torch.float32).reshape(-1,1),torch.tensor(Y,dtype=torch.float32).reshape(-1,1)),dim=1))


# Reshape samples_u to match the grid
samples_u_grid = samples_u.reshape(X.shape)

# print("Z3 shape = ", Z3.shape)

# Z3_np_reshaped = Z3.detach().numpy()
# print("the shape of the colors is ", Z3.detach().numpy().shape)
# print("Z3_np reshaped shape: ", Z3_np_reshaped.shape)

# Reshape Z3_np to [10000]

# print("Z3.detach().numpy() shape", Z3.detach().numpy().shape)

# plt.figure(3)
# plt.subplot(1,3,1)
# plt.scatter(fine_samples_x_test.reshape(-1,1).detach().numpy(),fine_samples_y_test.reshape(-1,1).detach().numpy(),c=Z3.detach().numpy())
# plt.colorbar(norm=norm)
# plt.title("Fine Trained Model Prediction on Test Data")
#
# plt.subplot(1,3,2)
# samples_u_test = np.array(u_true(fine_samples_x_test,fine_samples_y_test))
# samples_u_test=samples_u_test.reshape((-1,1))
# # print("samples_u_test shape: ", samples_u_test.shape)
# # print("samples u test shape ", samples_u_test.shape)
# # print("Z3 shape" , Z3.shape)
#
# plt.scatter(fine_samples_x_test.reshape(-1,1).detach().numpy(), fine_samples_y_test.reshape(-1,1).detach().numpy(), c=samples_u_test)
# plt.colorbar(norm=norm)
# plt.title("True solution on Test Data")
#
# plt.subplot(1,3,3)
# plt.scatter(fine_samples_x_test.reshape(-1,1).detach().numpy(), fine_samples_y_test.reshape(-1,1).detach().numpy(),c = Z3_np_reshaped-samples_u_test)
# plt.colorbar(norm=norm)
# plt.title("Difference Between Model and True Solution on Test Data")
# plt.show()

# Create subplots with adjusted layout
fig, axs = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1,1,1]})

# Plot 1: Fine Trained Model Prediction on Fine Data
im1=axs[0].imshow(Z2[:,0].detach().numpy().reshape((100, 100)))
axs[0].set_title("Fine Trained Model Prediction on Fine Data",fontsize=10)
# cbar_ax1 = fig.add_axes([0.1, 0.15, 0.02, 0.7])
fig.colorbar(plt.imshow(Z2[:,0].detach().numpy().reshape((100, 100))), ax=axs[0], norm=norm,pad=0.2, shrink=0.6)

# Plot 2: True solution on Training data
im2=axs[1].imshow(samples_u_grid, extent=(fine_train_data_x_with_bdd.min(), fine_train_data_x_with_bdd.max(), fine_train_data_y_with_bdd.min(), fine_train_data_y_with_bdd.max()), cmap='viridis')
axs[1].set_title("True solution on Training data",fontsize=10)
# cbar_ax2 = fig.add_axes([0.45, 0.15, 0.02, 0.7])
fig.colorbar(plt.imshow(samples_u_grid, cmap='viridis'), ax=axs[1], norm=norm,pad=0.2, shrink=0.6)

# Plot 3: Difference between model and true solution
axs[2].imshow(Z2[:,0].detach().numpy().reshape((100, 100)) - np.asarray(samples_u_grid))
axs[2].set_title("Difference between model and true solution",fontsize=10)
# cbar_ax3 = fig.add_axes([1.2, 0.15, 0.02, 0.7])
fig.colorbar(plt.imshow(Z2[:,0].detach().numpy().reshape((100, 100)) - np.asarray(samples_u_grid)), ax=axs[2], norm=norm,pad=0.2, shrink=0.6)

plt.tight_layout()
# plt.show()
plt.savefig("2nd_order_discontinuous_coarse_guess.png")




#Put the error computation using an integral of int(u_true-u_theta)

#Ask Alessandro: The error calculation is giving me a negative number
#but I'm trying to integrate (u_true - u_theta)^2, so I don't know why it's negative
# Z3_reshaped = torch.squeeze(Z3, axis=1)
# samples_u_test = torch.tensor(samples_u_test)

# z_values = torch.abs(Z3_reshaped.detach() - torch.squeeze(samples_u_test).detach())
# print("Z3 reshaped shape ",Z3_reshaped.detach().shape)
# print("samples u test shape ",samples_u_test.detach().shape)
# z_values_squared = torch.square(z_values)
# print("Z values squared look like ", z_values_squared)
# print("z_vaules_squared shape: ",z_values_squared.shape)
# from scipy.integrate import simps

# print("z values shape: ", z_values.shape)

# Assuming fine_samples_y_test and fine_samples_x_test are PyTorch tensors
# Convert them to NumPy arrays before integration
y_values = torch.tensor(np.arange(domain_y[0],domain_y[1],200))#fine_samples_y_test.detach().numpy()
x_values = torch.tensor(np.arange(domain_x[0],domain_x[1],200))#fine_samples_x_test.detach().numpy()


from scipy import interpolate
import scipy.integrate as spi


#This is how I should call u_true on my evalulation points
print(u_true(torch.tensor([0,0]).detach().numpy()[0],torch.tensor([0,0]).detach().numpy()[1]))

evaluation_points = torch.zeros(7,2)

aux1, aux2 = 1/np.sqrt(3), np.sqrt(3/5)
aux3 = np.sqrt(14/15)


#Define evaluation points
evaluation_points[0] = torch.tensor([0, 0])
evaluation_points[1] = torch.tensor([aux1, aux2])
evaluation_points[2] = torch.tensor([-aux1, aux2])
evaluation_points[3] = torch.tensor([aux1, -aux2])
evaluation_points[4] = torch.tensor([-aux1, -aux2])
evaluation_points[5] = torch.tensor([aux3, 0])
evaluation_points[6] = torch.tensor([-aux3, 0])

#Call the model and the true solution and compute |true - predicted| and integrate that
true_model_diff = []
for i in range(evaluation_points.shape[0]):

  u_val = u_true(evaluation_points[i][0],evaluation_points[i][1])

  model_fine_val = model_fine(evaluation_points[i])

  true_model_diff.append((u_val-model_fine_val)**2)


true_model_diff = [difference.item() for difference in true_model_diff]

#Define the weights
weights = np.array([8/7, 20/36, 20/36, 20/36, 20/36, 20/63, 20/63])


residual = 0
for i in range(evaluation_points.shape[0]):
  residual += weights[i]*true_model_diff[i]

print("Integrated error is ",np.sqrt(residual))


print("maximum magnitude error = ", np.max(np.abs(Z2[:,0].detach().numpy()-np.squeeze(samples_u.detach().numpy().reshape(-1,1)))))
print("mean magnitude error: ",np.mean(np.abs(Z2[:,0].detach().numpy()-np.squeeze(samples_u.detach().numpy().reshape(-1,1)))))

max_error = np.max(np.abs(Z2[:,0].detach().numpy()-np.squeeze(samples_u.detach().numpy().reshape(-1,1))))
mean_error = np.mean(np.abs(Z2[:,0].detach().numpy()-np.squeeze(samples_u.detach().numpy().reshape(-1,1))))


with open("output.txt",'a') as f:
  f.write("\n"+"2nd Order Discontinuous Coarse Initialization Fine Training: "+str(toc-tic)+str(" seconds to train"))
  f.write("\n"+"Max error: " + str(max_error))
  f.write("\n"+"Mean error: " + str(mean_error))
  f.write("\n"+"\n"+str("L^2 error ")+str(np.sqrt(residual))+"\n")

# Concatenate your input data along dimension 1
concatenated_data = torch.cat((fine_train_data_x_with_bdd, fine_train_data_y_with_bdd), dim=1)

# Pass concatenated data through the model to get predictions
Z2 = model_fine(concatenated_data)

# Now you can print the prediction
print("Z2:", Z2)




# upper_bound_for_error = 4*np.max(np.abs(Z2[:,0].detach().numpy()-np.squeeze(samples_u.detach().numpy())))
# print("upper bound for error should be ", upper_bound_for_error)
# print("This is based on the fact that the upper bound for an integral is (size of set)*(maximum value in that set)")
# print(type(interior_losses))
interior_losses = [loss.detach() for loss in interior_losses]
# print(type(boundary_losses))
boundary_losses = [loss.detach() for loss in boundary_losses]
# print(type(np.array(range(1,len(interior_losses)+1))))
# print(type(np.array(range(1,len(boundary_losses)+1))))
# print(len(interior_losses))


# plt.scaplt.scatter(range(1,len(interior_losses)+1), np.array(interior_losses))
# plt.title("Interior Losses")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
#
# plt.show()
#
#
# plt.scatter(range(1,len(boundary_losses)+1), np.array(boundary_losses))
# plt.title("Boundary Losses")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")tter(range(1,len(interior_losses)+1), np.array(interior_losses))
# plt.title("Interior Losses")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
#
# plt.show()
#
#
# plt.scatter(range(1,len(boundary_losses)+1), np.array(boundary_losses))
# plt.title("Boundary Losses")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")

plt.clf()
boundary_losses = [i.item() for i in boundary_losses]
# print(boundary_losses)
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1,len(boundary_losses)+1),np.array(boundary_losses))
plt.title("Boundary Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("2nd_order_discont_10x10_coarse_guess_bdd_losses")
# plt.show()

plt.clf()
interior_losses = [i.item() for i in interior_losses]
# print(interior_losses)
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1,len(interior_losses)+1),np.array(interior_losses))
plt.title("Interior Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("2nd_order_discont_10x10_coarse_guess_int_losses")
#
# plt.show()