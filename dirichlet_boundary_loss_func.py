from nn import *
from problem import *

import torch

def dirichlet_boundary_loss_func(x,y,g,model):

  torch.set_default_dtype(torch.float32)
  
  pred = model(torch.cat((x,y),dim=1))
 
  pred = torch.squeeze(pred)
 
  g_val = torch.squeeze(g_true(x,y))
  # print("in dirichlet_boundary_loss_func")
  # print("pred shape is ", pred.shape)
  # print("g_val shape is ", g_val.shape)
  
  return (torch.norm(pred-g_val)**2)