from nn_fine import *
from dirichlet_train_data_generation import *
from dirichlet_boundary_loss_func import *
from neumann_boundary_loss_func import *
from neumann_train_data_generation import *
from interior_loss_func import *
from all_train_data import *
from samples_f import *
from samples_g import *
from weights import *
from coarse_samples_u import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from fem import data_points#*

# print("DATA POINTS: ", data_points)

# Function to print model parameters
def print_model_params(model, tag):
    print(f"{tag} model parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")

# global data_points

torch.autograd.set_detect_anomaly(True)

domain_x = [-1,1]
domain_y = [-1,1]


#Define the boundary points
train_data_boundary_x = pd.read_csv("train_data_boundary_x.csv")
train_data_boundary_x = torch.tensor(train_data_boundary_x.iloc[:,1])

train_data_boundary_y = pd.read_csv("train_data_boundary_y.csv")
train_data_boundary_y = torch.tensor(train_data_boundary_y.iloc[:,1])

#Define the interior
fine_samples_x, fine_samples_y = np.mgrid[domain_x[0]:domain_x[1]:100j, domain_y[0]:domain_y[1]:100j]
fine_samples_x = torch.tensor(fine_samples_x)

fine_samples_y = torch.tensor(fine_samples_y)
fine_xy = np.vstack((fine_samples_x.flatten(), fine_samples_y.flatten())).T


#Convert the fine data into tensors
fine_samples_x = torch.tensor(fine_xy[:,0])
fine_samples_y = torch.tensor(fine_xy[:,1])

fine_samples_x.requires_grad = True
fine_samples_y.requires_grad = True





#Concatenate interior data with boundary data
fine_train_data_x_with_bdd = torch.cat((torch.tensor(fine_samples_x.unsqueeze(1)), torch.tensor(train_data_boundary_x).unsqueeze(1)), axis=0)
fine_train_data_y_with_bdd = torch.cat((torch.tensor(fine_samples_y.unsqueeze(1)), torch.tensor(train_data_boundary_y).unsqueeze(1)), axis=0)


fine_xy = np.hstack((fine_samples_x.detach().reshape(-1,1),fine_samples_y.detach().reshape(-1,1)))



indices_to_remove = np.logical_or(np.logical_or(fine_xy[:, 0] == 1, fine_xy[:, 0] == -1),
                                   np.logical_or(fine_xy[:, 1] == 1, fine_xy[:, 1] == -1))

# Remove rows with those indices
fine_xy_interior = fine_xy[~indices_to_remove]


#Define the network to train on fine data
model_fine = NeuralNetwork_fine()



torch.set_default_dtype(torch.float32)

norm = mpl.colors.Normalize(vmin=0, vmax=6)
plt.figure(figsize = (15,5))

fine_train_data_x_with_bdd= torch.tensor(fine_train_data_x_with_bdd,dtype=torch.float32)
fine_train_data_y_with_bdd= torch.tensor(fine_train_data_y_with_bdd,dtype=torch.float32)

print("(torch.tensor(data_points[:,0]) shape", (torch.tensor(data_points[:,0]).shape))
print("(torch.tensor(data_points[:,1]) shape", (torch.tensor(data_points[:,1]).shape))

interior_data_x = torch.tensor(data_points[:, 0].reshape(-1, 1)).to(torch.float32)
interior_data_y = torch.tensor(data_points[:, 1].reshape(-1, 1)).to(torch.float32)



print("interior data x dtype", interior_data_x.dtype)
print("interior data y dtype", interior_data_y.dtype)

# model_input=torch.cat((interior_data_x,interior_data_y),dim=1)


# print((torch.cat((torch.tensor(data_points[:,0]).reshape(-1,1),torch.tensor(data_points[:,1]).reshape(-1,1)),dim=1).shape))



#This line runs fine when I put the breakpoint right after it. But when I try to run the whole file,
#the traceback of the error goes to here! What does this mean?
Z = model_fine(torch.cat((interior_data_x,interior_data_y),dim=1))
# breakpoint()

#Call f on all of the fine # interior data
samples_f = f_true(interior_data_x,interior_data_y)



print("fine_train_data_x_with_bdd shape", fine_train_data_x_with_bdd.shape)

optimizer = torch.optim.LBFGS(model_fine.parameters(),max_iter=70,line_search_fn='strong_wolfe')



losses=[]
boundary_losses = []
interior_losses = []
neumann_bdd_losses = []
dirichlet_bdd_losses = []
data_fit_losses = []
physics_losses = []

num_epochs = 150

# interior_physics_loss = 0
torch.set_default_dtype(torch.float32)


data_points_tensor = torch.tensor(data_points, requires_grad=True)
x = torch.tensor(data_points_tensor[:, 0]).unsqueeze(dim=1).to(torch.float32)
y = torch.tensor(data_points_tensor[:, 1]).unsqueeze(dim=1).to(torch.float32)
x.requires_grad=True
y.requires_grad=True

def closure_fine():
    torch.set_default_dtype(torch.float32)

    optimizer.zero_grad()


    # Call fine trained model
    output_full_interior = model_fine(torch.tensor(fine_xy))

    # FIX THIS
    #This should compare the samples_u to


    # print("Z shape: ", Z.shape)
    # print("samples_u shape: ", torch.tensor(samples_u).reshape(-1,1).shape)
    # print("BOOPPPPPP")
    #
    # print("DATA FIT INTERIOR LOSS TERM SHAPES")
    # print ("Z shape",Z.shape)
    # print("torch.tensor(samples_u).reshape(-1,1) shape", torch.tensor(samples_u).reshape(-1,1).shape)

    samples_g_train = g_true(torch.tensor(dirichlet_boundary[:, 0],dtype=torch.float32), torch.tensor(dirichlet_boundary[:, 1],dtype=torch.float32))


    inputs = torch.cat((torch.tensor(x), torch.tensor(y)), dim=1).to(torch.float32)
    inputs.requires_grad=True
    inputs.retain_grad()
    u = model_fine(inputs)



    # Check if the model has parameters that require gradients
    if not any(param.requires_grad for param in model_fine.parameters()):
        print("Model has no trainable parameters!")
    criterion = torch.nn.MSELoss()
    # print("u.float()", u.float())
    # print("torch.tensor(data_points[:,2],dtype=torch.float32)", torch.tensor(data_points[:,2],dtype=torch.float32))
    # print("u float shape ", u.float().shape)
    # print("torch.tensor(data_points[:,2],dtype=torch.float32) shape", torch.tensor(data_points[:, 2], dtype=torch.float32).shape)

    # loss = criterion(u.float(), torch.tensor(data_points[:,2],dtype=torch.float32).unsqueeze(dim=1))

    # Check if gradients are being computed for the loss
    # if loss.grad_fn is None:
    #     print("No gradients are being computed for the loss!")

    # print("The loss is ", loss)

    # print("Just did loss.backward(). Is that the None?")
    # print(inputs.grad)
    # samples_u_float32 = torch.tensor(samples_u).reshape(-1, 1).to(torch.float32)
    data_fit_interior_loss = criterion(Z,samples_u.reshape(-1, 1))



    boundary_loss = dirichlet_boundary_loss_func(
        dirichlet_boundary[:, 0].unsqueeze(dim=1),
        dirichlet_boundary[:, 1].unsqueeze(dim=1),
        samples_g_train.unsqueeze(dim=1),
        model_fine).to(torch.float32)



    interior_physics_loss = interior_loss_func(data_points_tensor[:, 0].unsqueeze(dim=1),
                                               data_points_tensor[:, 1].unsqueeze(dim=1),
                                               data_points_tensor[:, 2].unsqueeze(dim=1), model_fine).to(torch.float32)



    # print("interior phys loss: ", interior_physics_loss)
    # print("data_fit_interior_loss: ", data_fit_interior_loss)
    #
    # print("bdd loss shape: ", boundary_loss)

    print("INTERIOR PHYSICS LOSS: ", interior_physics_loss)

    print("BOUNDARY LOSS: ", dirichlet_boundary_loss_func(
            dirichlet_boundary[:, 0].unsqueeze(dim=1),
            dirichlet_boundary[:, 1].unsqueeze(dim=1),
            samples_g_train.unsqueeze(dim=1),
            model_fine))

    # total_loss = interior_physics_loss+boundary_loss+data_fit_interior_loss

    print("sum of the losses is: ", interior_physics_loss+boundary_loss+data_fit_interior_loss)

    # breakpoint()

    # torch.tensor(data_points).requires_grad = True

    # print("THE TYPE OF DATA POINTS IS ", type(data_points))


    # interior_physics_loss2 = interior_physics_loss.clone().to(torch.float32)
    #
    # boundary_loss2 = boundary_loss.clone().to(torch.float32)
    #
    # data_fit_interior_loss2 = data_fit_interior_loss.clone().to(torch.float32)

    #Ultimately, I want the objective to be interior_physics_loss+boundary_loss+data_fit_interior_loss,
    #But just try isolating each one for now
    # breakpoint()
    print("data fit interior loss float 32 dtype: ", (data_fit_interior_loss.to(torch.float32)).dtype)
    objective = interior_physics_loss+boundary_loss+float(data_fit_interior_loss)#+data_fit_interior_loss.to(torch.float32)#torch.tensor(interior_physics_loss,requires_grad=True)#interior_loss_func_weight_fine * interior_physics_loss2 + data_fit_dirichlet_bdd_weight * boundary_loss2 + data_fit_interior_weight_fine * data_fit_interior_loss2

    print(objective)

    # breakpoint()

    objective.backward(retain_graph=True)

    # breakpoint()
    # print("just backpropagated")





    # objective.backward(retain_graph=True,allow_unused=True)
    # print("THIS IS THE STORY OF A GRADIENT")
    # print(x.grad)

    return objective

torch.set_default_dtype(torch.float32)

import timeit
tic = timeit.default_timer()


#CLEAN UP AFTER HERE

print("About to start minimizing")
for epoch in range(num_epochs):


  # Print model parameters before optimization step
  # print_model_params(model_fine, "Before optimization")

  parameters = [p for p in model_fine.parameters() if p.grad is not None and p.requires_grad]

  # print("just defined parameters")





  # total_norm = 0
  # # print("This is where the problem is?")
  # for p in parameters:
  #   param_norm = p.grad.detach().data.norm(2)
  #   total_norm = total_norm + param_norm.item() ** 2
  # total_norm = total_norm ** 0.5

  n = torch.tensor(dirichlet_boundary).shape[0]#+torch.tensor(neumann_boundary).shape[0]

  #Set batch size
  interior_batch_size = 20

  #boundary_batch_size = int(n/(int(train_data.shape[0]/interior_batch_size)))

  # print("BEFORE FILTERING: Train data is ", train_data)

  # plt.scatter(train_data[:,0], train_data[:,1])
  # plt.title("ARE ALL THESE ON THE BOUNDARY?")
  # plt.show()

  # print("fine_xy_interior[:, 0] shape", fine_xy_interior[:, 0].shape)
  # breakpoint()
  #This should actually be called on the data from the finite element solution

  # breakpoint()

  #This line is where the issue is, but every line IN the closure is working.
  #So I'm not sure WHAT the problem is

  optimizer.step(closure_fine)

  # Print model parameters after optimization step
  # print_model_params(model_fine, "After optimization")

  #Compute the physics loss on the interior




  #Randomly order the indices of the boundary data points
  idx_bdd = torch.randperm(n)

  idx_int = torch.randperm(train_data.shape[0])


  #Evaluate the model on the training data
  fine_xy = torch.tensor(fine_xy,dtype=torch.float32)
  output_interior = model_fine(torch.tensor(fine_xy))
  output_dirichlet = model_fine(torch.tensor(dirichlet_boundary))
  # output_neumann = model_fine(torch.tensor(neumann_boundary))


  # dirichlet_boundary_weight = 100
  # neumann_boundary_weight = 50
  # interior_loss_weight = 0
  # data_fit_weight = 10

  # dirichlet_loss =  data_fit_dirichlet_bdd_weight * dirichlet_boundary_loss_func(dirichlet_boundary[:,0].unsqueeze(dim=1),
  #                                              dirichlet_boundary[:,1].unsqueeze(dim=1),
  #                                              samples_g_train.unsqueeze(dim=1),model_fine)

  # neumann_loss = neumann_boundary_weight * neumann_boundary_loss_func(torch.tensor(neumann_boundary)[:,0].unsqueeze(dim=1),torch.tensor(neumann_boundary)[:,1].unsqueeze(dim=1),model_fine)


  # boundary_loss = data_fit_dirichlet_bdd_weight*dirichlet_loss #+ neumann_boundary_weight*neumann_loss




  # print("OUTPUT INTERIOR SHAPE ",output_interior.shape)

  data_fit_loss_dirichlet = torch.norm(output_dirichlet-torch.reshape(u_true(dirichlet_boundary[:,0],dirichlet_boundary[:,1]),(-1,1)))

  boundary_losses.append(data_fit_loss_dirichlet)#+data_fit_loss_neumann)

  # interior_losses.append(interior_physics_loss)


  # dirichlet_bdd_losses.append(dirichlet_loss)



  #Keepin' you in the loop about how your training is going

  if epoch % 100 == 0:
      # print("boop")
      print('Epoch' + str(epoch))
      # print('total: ', losses[-1])
      # print('Data fit loss interior: ', interior_losses[-1].item())
      # print('Data fit loss boundary: ', boundary_losses[-1].item())
      # print('Data fit: ', torch.norm(output_interior - torch.reshape(u_true(coarse_xy[:, 0], coarse_xy[:, 1]), (-1, 1))))
      # print('Data fit loss Dirichlet', dirichlet_bdd_losses[-1].item())
      # print('Data fit loss Neumann', neumann_bdd_losses[-1].item())
      # print("interior_physics_loss", interior_physics_loss)
      # print('Interior loss: ',interior_physics_loss)
      # print('Physics Loss', interior_losses[-1].item())

toc = timeit.default_timer()

print("Training took "+str(toc-tic)+str(" seconds"))

# print("LET'S SEE SOME GRADIENTS")
#
# for param in model_fine.parameters():
#     print(param.grad)


#After the second training

# Z2 = model_fine(torch.cat((fine_train_data_x_with_bdd,fine_train_data_y_with_bdd),dim=1))

#Print parameters after the second training

# print("ROUND 2 PARAMETERS")
# for param in model_fine.parameters():
#   # print(param.data.dtype)
#   print("CHECKING FOR GRADIENTS BEING NONE")
#   print(param.grad is not None)

#THIS PLOT ISN'T SHOWING THE RIGHT THING. IT'S SHOWING THE EXACT SAME THING AS THE FIRST PLOT
#BEFORE THE SECOND TRAINING

# print("shape of Z2[:,0].detach().numpy() is ", Z2[:,0].detach().numpy().shape)
# print("shape of fine_train_data_x_with_bdd.detach().numpy() is ", fine_train_data_x_with_bdd.detach().numpy().shape)



x_grid = np.linspace(-1,1,100)
y_grid = np.linspace(-1,1,100)
X, Y = np.meshgrid(x_grid, y_grid)

# samples_u = u_true(X,Y)

# Z2 = model_fine(torch.cat((torch.tensor(X,dtype=torch.float32).reshape(-1,1),torch.tensor(Y,dtype=torch.float32).reshape(-1,1)),dim=1))


# Reshape samples_u to match the grid
print("Samples u shape", samples_u.shape)
samples_u_grid = (samples_u.reshape(-1,1)).reshape((11,11))


from matplotlib.colors import Normalize

# breakpoint()

# Create subplots with adjusted layout
fig, axs = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1,1,1]})

# Plot 1: Fine Trained Model Prediction on Fine Data
im1=axs[0].imshow(Z[:,0].detach().numpy().reshape((int(Z[:,0].shape[0]**.5), int(Z[:,0].shape[0]**.5))))
axs[0].set_title("Fine Trained Model Prediction on Fine Data",fontsize=10)
# cbar_ax1 = fig.add_axes([0.1, 0.15, 0.02, 0.7])
fig.colorbar(plt.imshow(Z[:,0].detach().numpy().reshape((11, 11)), ax=axs[0], norm=norm,pad=0.2, shrink=0.6))

# fig.colorbar(plt.imshow(Z[:,0].detach().numpy().reshape((11,11)), ax=axs[0], norm=norm,pad=0.2, shrink=0.6))

# Plot 2: True solution on Training data
im2=axs[1].imshow(samples_u_grid, extent=(fine_train_data_x_with_bdd.min(), fine_train_data_x_with_bdd.max(), fine_train_data_y_with_bdd.min(), fine_train_data_y_with_bdd.max()), cmap='viridis')
axs[1].set_title("True solution on Training data",fontsize=10)
# cbar_ax2 = fig.add_axes([0.45, 0.15, 0.02, 0.7])
fig.colorbar(plt.imshow(samples_u_grid, cmap='viridis'), ax=axs[1], norm=norm,pad=0.2, shrink=0.6)

# Plot 3: Difference between model and true solution
axs[2].imshow(Z[:,0].detach().numpy().reshape((100, 100)) - np.asarray(samples_u_grid))
axs[2].set_title("Difference between model and true solution",fontsize=10)
# cbar_ax3 = fig.add_axes([1.2, 0.15, 0.02, 0.7])
fig.colorbar(plt.imshow(Z[:,0].detach().numpy().reshape((11, 11)) - np.asarray(samples_u_grid)), ax=axs[2], norm=norm,pad=0.2, shrink=0.6)

plt.tight_layout()
# plt.show()
plt.savefig("2nd_order_discont_FEM_guess.png")


max_error = np.max(np.abs(Z[:,0].detach().numpy()-np.squeeze(samples_u.detach().numpy().reshape(-1,1))))
mean_error = np.mean(np.abs(Z[:,0].detach().numpy()-np.squeeze(samples_u.detach().numpy().reshape(-1,1))))

print("maximum magnitude error = ", np.max(np.abs(Z[:,0].detach().numpy()-np.squeeze(samples_u.detach().numpy().reshape(-1,1)))))
print("mean magnitude error = ",np.mean(np.abs(Z[:,0].detach().numpy()-np.squeeze(samples_u.detach().numpy().reshape(-1,1)))))
# plt.show()

#Plot boundary and interior losses
# print(boundary_losses)
plt.clf()
boundary_losses = [i.item() for i in boundary_losses]
# print(boundary_losses)
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1,len(boundary_losses)+1),np.array(boundary_losses))
plt.title("Boundary Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("2nd_order_discont_fem_guess_bdd_losses")
# plt.show()

plt.clf()
interior_losses = [i.item() for i in interior_losses]
# print(interior_losses)
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1,len(interior_losses)+1),np.array(interior_losses))
plt.title("Interior Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("2nd_order_discont_fem_guess_int_losses")
# plt.show()

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
  residual = residual + weights[i]*true_model_diff[i]

print("Integrated error is ",np.sqrt(residual))


with open("output.txt",'a') as f:
    f.write("\n"+"max error: "+str(max_error))
    f.write("\n"+"mean error: "+str(mean_error))
    f.write("\n"+"2nd Order Discontinuous FEM Initialized Fine Training: "+str(toc-tic)+str(" seconds to train"))
    f.write("\n"+"\n"+str("L^2 error ")+str(np.sqrt(residual))+"\n")

