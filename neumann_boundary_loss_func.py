from problem import * 

def neumann_boundary_loss_func(x,y,model):

  torch.set_default_dtype(torch.float32)
  x.requires_grad=True
  y.requires_grad=True
  #Define model. Store data in an n x 2 array where each row is a data point
  u = model(torch.cat((x,y),dim=1))
  
  #Take derivatives that appear in the heat equation
  u_x = torch.autograd.grad(
      u,x,
      grad_outputs=torch.ones_like(u),
      retain_graph=True,
      create_graph=True
  )[0]

  u_y = torch.autograd.grad(
      u,y,
      grad_outputs=torch.ones_like(u),
      retain_graph=True,
      create_graph=True
  )[0]

  return torch.norm(u_y-neumann_bdd(x,y))**2
