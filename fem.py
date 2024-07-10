from __future__ import print_function

from fenics import *

import matplotlib.pyplot as plt

from coarse_samples_u import *

from coarse_data_generation import *
import torch


# Create mesh and define function space

# Create mesh and define function space

nx = ny = 10

mesh = RectangleMesh(Point(-1, -1), Point(1, 1), nx, ny)

V = FunctionSpace(mesh, 'P', 2)


# Define boundary condition

u_D = Expression('(x[0]-x[1])*(x[0]-x[1])/2.*(x[0]<x[1])', degree=2)


def boundary(x, on_boundary):

    return on_boundary


bc = DirichletBC(V, u_D, boundary)


# Define variational problem

u = TrialFunction(V)

v = TestFunction(V)

f = Expression('-2*(x[0]<x[1])',degree=1)

a = dot(nabla_grad(u), nabla_grad(v))*dx

L = f*v*dx


equation = inner(nabla_grad(u), nabla_grad(v))*dx == f*v*dx


prm = parameters["krylov_solver"] # short form

prm["absolute_tolerance"] = 1E-10

prm["relative_tolerance"] = 1E-6

prm["maximum_iterations"] = 1000

#set_log_level(DEBUG)

set_log_level(LogLevel.ERROR)




# Compute solution

u = Function(V)

solve(equation, u, bc, solver_parameters={"linear_solver": "cg", "preconditioner": "ilu"})

info(parameters,True)

# Plot solution and mesh

plot(u)

plot(mesh)

print('Mesh')

#print(str(mesh))


# Save solution to file in VTK format

vtkfile = File('poisson/BenSolution.pvd')

vtkfile << u


# Compute error in L2 norm

error_L2 = errornorm(u_D, u, 'L2')


# Compute maximum error at vertices

vertex_values_u_D = u_D.compute_vertex_values(mesh)

vertex_values_u = u.compute_vertex_values(mesh)

import numpy as np

error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))


coor = mesh.coordinates()

data_points = np.zeros((mesh.num_vertices(),3))

if mesh.num_vertices() == len(vertex_values_u):

  for i in range(mesh.num_vertices()):
    data_points[i,:] = [coor[i][0], coor[i][1], vertex_values_u[i]]
    # print(coor[i][0], coor[i][1], vertex_values_u[i])

print("data_points[:,0] shape: ", data_points[:,0].shape)
# print("Look here!")
x_vals = np.linspace(-1,1,10)
y_vals = np.linspace(-1,1,10)
print(u(0.,0.))
u_fem_np = np.zeros((mesh.num_vertices(),1))
for i in range(len(x_vals)):
    u_fem_np[i] = u(x_vals[i],y_vals[i])
    # print(u(x_vals[i],y_vals[i]))
# print(u(x_vals,y_vals))
u_fem = torch.tensor(u_fem_np,requires_grad=True)

# Print errors
print(u_fem.dtype)
print('error_L2  =', error_L2)

print('error_max =', error_max)


# Hold plot

plt.show()
