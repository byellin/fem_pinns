from coarse_data_generation import *

def on_boundary(x,y):
  if x==domain_x[0] or x==domain_x[1] or y==domain_y[0] or y==domain_y[1]:
    return True
  else:
    return False

def on_right_boundary(x,y):
  if x==domain_x[1]:
    return True 
  else:
    return False

#Fix this function!
def on_neumann_boundary(x,y):
  return False
  # if y==1 or y==-1:
  #   return True
  # else:
  #   return False

def on_dirichlet_boundary(x,y):
  return True
  # if x==1 or x==-1:
  #   return True
  # else:
  #   return False

#Number of points to pick on the boundary
k=10


print("boundary_check.py lookin' good ")