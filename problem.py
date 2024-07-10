import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import timeit

#Checkerboard solution

#Set Dirichlet boundaries on the left and right
#Set Neumann boundaries on the top and bottom 



torch.set_default_dtype(torch.float32)


def u_true(x,y):

	#Ex. 1
	# return torch.sin(torch.pi*torch.tensor(x.clone().detach()))
	
	#Ex. 2 
	#return torch.sin(torch.pi * torch.tensor(x.clone().detach())) * torch.cos(torch.pi * torch.tensor(y.clone().detach()))

	#Ex. 3
	return (torch.tensor(x)-torch.tensor(y))*(torch.tensor(x)-torch.tensor(y))/2.0*(x<y)


def f_true(x,y):
	alpha = 1
	#Ex. 1
	# return torch.pi**2*torch.sin(torch.pi*torch.tensor(x.clone().detach()))
	
	#Ex. 2
	#return 2*(torch.pi ** 2) * torch.sin(torch.pi * torch.tensor(x).clone().detach()) * torch.cos(torch.pi * torch.tensor(y).clone().detach())

	#Ex. 3
	return -2.0*(x<y)

	#Ex. 4
	# return torch.exp(-alpha*torch.tensor(t))*torch.sin(torch.tensor(x))
	
def g_true(x,y):

	#Ex. 1 
	return u_true(torch.tensor(x.clone().detach()),torch.tensor(y.clone().detach()))

	#Ex. 2
	# return u_true(torch.tensor(x), torch.tensor(y))

	#Ex. 3
	# return u_true(torch.tensor(x), torch.tensor(y))

	#Ex.4 
	# return torch.zeros_like(x)

def neumann_bdd(x,y):


	#Ex.1 
	#return 0 #Is this even the right neumann bc?
	
	#Ex. 2
	# return -torch.pi*torch.sin(torch.pi*x)*torch.sin(torch.pi*y)

	#Ex. 3 FILL THIS IN

	return 0

#Set Dirichlet boundaries on the top and bottom 
#Set Neumann boundaries on the bottom and right

# u_true = lambda x, y: torch.sin(torch.pi * torch.tensor(x).clone().detach()) * torch.cos(torch.pi * torch.tensor(y).clone().detach())
# f_true = lambda x, y: -2 * (torch.pi ** 2) * torch.sin(torch.pi * torch.tensor(x).clone().detach()) * torch.cos(torch.pi * torch.tensor(y).clone().detach())
# g_true = lambda x, y: u_true(torch.tensor(x), torch.tensor(y))
# neumann_bdd = lambda x,y: -torch.pi*torch.sin(torch.pi*x)*torch.sin(torch.pi*y)


#Discontinuous solution


#Uncomment these standardly defined functions when I want to switch problems
# def u_true(x,y):
# 	return (torch.tensor(x)-torch.tensor(y))*(torch.tensor(x)-torch.tensor(y))/2.0*(x<y)

# def f_true(x,y): 
# 	return 2.0*(x<y)

# def g_true(x,y):
# 	return u_true(torch.tensor(x), torch.tensor(y))

# def neumann_bdd(x,y):
# 	return 0#-torch.pi*torch.sin(torch.pi*x)*torch.sin(torch.pi*y)




# u_true = lambda x,y: (torch.tensor(x)-torch.tensor(y))*(torch.tensor(x)-torch.tensor(y))/2.0*(x<y)
# f_true = lambda x,y: 2.0*(x<y)
# g_true = lambda x,y: u_true(x,y)
# neumann_bdd = 0*x+0*y

