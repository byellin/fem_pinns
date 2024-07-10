#START HERE ON 6/23: RIGHT NOW, IT SAYS THAT MAT1 AND MAT2
#MUST HAVE THE SAME DTYPE


from nn import * 
from all_train_data import *
from neumann_train_data_generation import *
from dirichlet_train_data_generation import *
from dirichlet_boundary_loss_func import *
from neumann_boundary_loss_func import *
from interior_loss_func import *
from samples_f import *
from samples_g import *
from problem import *
from weights import *

print(model)


optimizer = torch.optim.LBFGS(model.parameters(),max_iter=70,line_search_fn='strong_wolfe')

#Initialize list of losses - keep track of interior and boundary losses separately
losses=[]
boundary_losses = []
interior_losses = []
neumann_bdd_losses = []
dirichlet_bdd_losses = []
data_fit_losses = []
physics_losses = []


num_epochs = 50 #Used to be 250 but I want to see if the fine training will improve it

def closure():
  torch.set_default_dtype(torch.float32)
  optimizer.zero_grad()


  # # data_fit_interior_weight = 0
  # data_fit_neumann_bdd_weight = 1
  # data_fit_dirichlet_bdd_weight = 10
  # data_fit_interior_weight = 100
  # interior_loss_func_weight_coarse = 100

  #Need to add data fitting term to my obejctive function (DO THIS IN THE AFTERNOON 3/22)

  #Call model on coarse data

  print("coarse_xy: ", coarse_xy)

  output_full_interior = model(torch.tensor(coarse_xy))
  # print(output_full_interior.shape)

  # output_full_neumann_boundary = model(torch.tensor(neumann_boundary))
  # print(output_full_neumann_boundary.shape)
  output_full_dirichlet_boundary = model(torch.tensor(dirichlet_boundary))
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
                                 model) +data_fit_interior_weight_coarse * data_fit_interior_loss + interior_loss_func_weight_coarse*interior_loss_func(torch.tensor(train_data[:,0]).unsqueeze(dim=1),torch.tensor(train_data[:,1]).unsqueeze(dim=1),samples_f,model)#+ data_fit_neumann_bdd_weight*neumann_boundary_loss_func(torch.tensor(neumann_boundary)[:,0].unsqueeze(dim=1),torch.tensor(neumann_boundary)[:,1].unsqueeze(dim=1),model)#

  #I need to add a term checking the data fitting on coarse_xy to my objective function. I don't think that's there yet


  # print("The Neumann boundary loss is ", neumann_boundary_loss_func(1,1))
  objective.backward(retain_graph=True)
  return objective

#Train neural network with physics!
torch.set_default_dtype(torch.float32)
import timeit
tic = timeit.default_timer()
for epoch in range(num_epochs):

  parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
  
  total_norm = 0
  
  for p in parameters:
    param_norm = p.grad.detach().data.norm(2)
    total_norm += param_norm.item() ** 2
  total_norm = total_norm ** 0.5
  # print("here's the dirichlet boundary")
  # print(dirichlet_boundary)
  # print("Inside train.py The length of the dirichlet boundary is ",len(dirichlet_boundary))
  n = torch.tensor(dirichlet_boundary).shape[0]

  #Set batch size
  interior_batch_size = 20

  #boundary_batch_size = int(n/(int(train_data.shape[0]/interior_batch_size)))

  optimizer.step(closure)

  #Randomly order the indices of the boundary data points
  idx_bdd = torch.randperm(n)

  idx_int = torch.randperm(train_data.shape[0])


  #Evaluate the model on the training data
  output_interior = model(torch.tensor(coarse_xy))
  output_dirichlet = model(torch.tensor(dirichlet_boundary))
  # output_neumann = model(torch.tensor(neumann_boundary))

 
  # data_fit_dirichlet_bdd_weight = 100
  # neumann_boundary_weight = 3
  # interior_loss_weight = 3
  # data_fit_weight = 10#50

  dirichlet_loss = data_fit_dirichlet_bdd_weight * dirichlet_boundary_loss_func(dirichlet_boundary[:,0].unsqueeze(dim=1),
                                              dirichlet_boundary[:,1].unsqueeze(dim=1),
                                              samples_g_train.unsqueeze(dim=1),model)

  # neumann_loss = neumann_boundary_weight * neumann_boundary_loss_func(torch.tensor(neumann_boundary)[:,0].unsqueeze(dim=1),torch.tensor(neumann_boundary)[:,1].unsqueeze(dim=1),model)


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


  data_fit_loss_interior = torch.norm(output_interior-torch.reshape(u_true(torch.tensor(coarse_xy[:,0]),torch.tensor(coarse_xy[:,1])),(-1,1)))
  # data_fit_loss_neumann = torch.norm(output_neumann-torch.reshape(u_true(neumann_train_data_boundary[:,0],neumann_train_data_boundary[:,1]),(-1,1)))
  data_fit_loss_dirichlet = torch.norm(output_dirichlet-torch.reshape(u_true(dirichlet_boundary[:,0],dirichlet_boundary[:,1]),(-1,1)))
  physics_loss = interior_loss_func(train_data[:,0].unsqueeze(dim=1),train_data[:,1].unsqueeze(dim=1),samples_f,model)

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
  physics_losses.append(physics_loss)
  #print(interior_loss)

  #Append the total loss (interior+boundary)
  #losses.append(boundary_loss)


  #Keepin' you in the loop about how your training is going
  if epoch%10==0:
    print('Epoch'+str(epoch))
    #print('total: ', losses[-1])
    print('Data fit loss interior: ', interior_losses[-1].item())

    #print('Data fit loss boundary: ', boundary_losses[-1].item())
    #print('Data fit: ', torch.norm(output_interior-torch.reshape(u_true(coarse_xy[:,0],coarse_xy[:,1]),(-1,1))))
    print('Data fit loss Dirichlet', dirichlet_bdd_losses[-1].item())
    # print('Data fit loss Neumann', neumann_bdd_losses[-1].item())
    print('Physics Loss', physics_losses[-1].item())

toc = timeit.default_timer()

print(toc-tic,str(" seconds to train"))

bestParams = model.state_dict()
torch.save({'state_dict': bestParams,}, 'coarse_result.pth')

print("Here are the parameters after the coarse training")
for parameter in model.parameters():
  print(parameter)
  print(parameter.shape)


# print(boundary_losses)
plt.clf()
boundary_losses = [i.item() for i in boundary_losses]
# print(boundary_losses)
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1,len(boundary_losses)+1),np.array(boundary_losses))
plt.title("Boundary Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("2nd_order_discont_10x10_coarse_train_bdd_losses")
# plt.show()

plt.clf()
interior_losses = [i.item() for i in interior_losses]
# print(interior_losses)
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1,len(interior_losses)+1),np.array(interior_losses))
plt.title("Interior Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("2nd_order_discont_10x10_coarse_train_int_losses")


with open("output.txt",'a') as f:
  f.write("\n"+"data_fit_dirichlet_bdd_weight: "+str(data_fit_dirichlet_bdd_weight))
  f.write("\n"+"data_fit_interior_weight_coarse: "+str(data_fit_interior_weight_coarse))
  f.write("\n"+"data_fit_interior_weight_fine: "+str(data_fit_interior_weight_fine))
  f.write("\n"+"interior_loss_func_weight_coarse: "+str(interior_loss_func_weight_coarse))
  f.write("\n"+"interior_loss_func_weight_fine: "+ str(interior_loss_func_weight_fine))
  f.write("\n"+"2nd Order Discontinuous Coarse Training: "+str(toc-tic)+str(" seconds to train"))
