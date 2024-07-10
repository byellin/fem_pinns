import torch
import numpy as np
from nn import *
from problem import *
from fem import data_points

#Define the loss function on the interior of the domain to calculate
#how closely the learned function solves the PDE you are trying to solve
def interior_loss_func(x,y,samples_f,model_fine):

    # print("ARE WE THERE YET?")
    # x=torch.tensor(x,requires_grad=True)
    # y=torch.tensor(y,requires_grad=True)

    # x.requires_grad=True
    # y.requires_grad=True

    # print("x version", x._version)
    # print("y version", y._version)

    #I THINK I MIGHT NEED TO CHANGE THE DTYPE OF THE U THAT GETS OUTPUTTED
    u = model_fine(torch.cat((x,y), dim=1))#.view(-1)

    # print('u version', u._version)


    # print("HERE U ARE")
    # print("u is ", u)
    # inputs = torch.cat((torch.tensor(x), torch.tensor(y)), dim=1)

    # print("inputs shape: ", inputs.shape)
    # inputs.requires_grad=True



    # breakpoint()

    # print('u shape: ', u.shape)

    n = len(u)

    # u = v[:,0]
    # sigma = v[:,1]

    # sigma = sigma_model(torch.cat((x,y),dim=1))

    # u = torch.squeeze(u)
    # sigma = torch.squeeze(sigma)

    # Take derivatives that appear in Laplace's equation

    # u_x = u.grad

    torch.autograd.set_detect_anomaly(True)
    # print("inputs[:,0] shape: ", inputs[:, 0].shape)
    # u_x = torch.autograd.grad(u, inputs[:, 0].unsqueeze(1), grad_outputs=torch.zeros_like(u), retain_graph=True, create_graph=True, allow_unused=True) #[0]



    # u_x = torch.zeros_like(u)

    # for i in range(len(u)):
    #     # u_i = u[i]
    #     u_x[i] =  torch.autograd.grad(u, inputs[i, 0], retain_graph=True,allow_unused=True)[0]

        # u_gradients[i]=u_i_grad
    # breakpoint()
    # print("u_x shape is:", u_x.shape)

    u2 = u.clone()

    u_x = torch.autograd.grad(u, x,
                              grad_outputs=torch.ones_like(u),
                              retain_graph=True,
                              create_graph=True,
                              allow_unused=True)[0]


    u_x_clone = u_x.clone()

    # print('u_x version', u_x_clone._version)

    # print('u_x is ', u_x)
    u_xx = torch.autograd.grad(
        u_x_clone, x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,

    )[0]

    u_xx_clone = u_xx.clone()

    # print("u_xx version", u_xx_clone._version)
    u_y = torch.autograd.grad(
        u2, y,
        grad_outputs=torch.ones_like(u2),
        retain_graph=True,
        create_graph=True,
        allow_unused=True
    )[0]

    # breakpoint()
    u_y_clone = u_y.clone()

    # print('u_y version', u_y_clone._version)

    # breakpoint()

    u_yy = torch.autograd.grad(
        u_y_clone, y,
        grad_outputs=torch.ones_like(u2),
        retain_graph=True,
        create_graph=True
    )[0]

    # breakpoint()

    u_yy_clone = u_yy.clone()

    # print("u_yy version", u_yy_clone._version)

    # breakpoint()


    # breakpoint()

    # u_x = torch.autograd.grad(
    #     u, inputs[:,0],
    #     grad_outputs=torch.ones_like(u),
    #     retain_graph=True,
    #     create_graph=True,
    #     allow_unused=True
    # )[0]
    #
    # print("u_x shape is ", u_x.shape)
    #
    # u_xx = torch.autograd.grad(
    #     u_x, inputs[:,0],
    #     grad_outputs=torch.ones_like(u),
    #     retain_graph=True,
    #     create_graph=True,
    #     allow_unused=True
    # )[0]
    #
    # u_y = torch.autograd.grad(
    #     u, inputs[:,1],
    #     grad_outputs=torch.ones_like(u),
    #     retain_graph=True,
    #     create_graph=True,
    #     allow_unused=True
    # )[0]
    #
    # u_yy = torch.autograd.grad(
    #     u_y, inputs[:,1],
    #     grad_outputs=torch.ones_like(u),
    #     retain_graph=True,
    #     create_graph=True,
    #     allow_unused=True
    # )[0]

    # print("the shape of u is ", u.shape)
    # print("And then we compute u_x")
    # u_x = torch.autograd.grad(
    #     u, torch.reshape(inputs[:,0],(-1,1)),
    #     grad_outputs=torch.ones_like(u),
    #     retain_graph=True,
    #     create_graph=True,
    #     allow_unused=True,
    #     materialize_grads=True
    # )[0]
    # print("u_x is ", u_x)
    # print("the shape of u_x is ", u_x.shape)
    # u_x.requires_grad=True
    # # u_x = torch.unsqueeze(u_x, dim=1)
    #
    #
    # # u_x.requires_grad=True
    #
    # # print("u_x shape: ", u_x.shape)
    #
    # # print("u_x shape: ", u_x.shape)
    #
    # print("inputs[:,0] shape", inputs[:,0].shape)
    #
    # u_x_new = torch.clone(u_x)
    #
    # print("u_x_new: ", u_x_new)
    #
    # u_xx = torch.autograd.grad(
    #     u_x_new, torch.clone(inputs[:,0]).unsqueeze(1),
    #     grad_outputs=torch.ones_like(u),
    #     retain_graph=True,
    #     create_graph=True,
    #     allow_unused=True
    # )[0]
    #
    # print("u_xx is ", u_xx)
    #
    # # u_xx = torch.unsqueeze(u_xx,dim=1)
    #
    # print("u_xx: ", u_xx)
    #
    # u_y = torch.autograd.grad(
    #     u, torch.reshape(inputs[:, 1], (-1, 1)),
    #     grad_outputs=torch.ones_like(u),
    #     retain_graph=True,
    #     create_graph=True,
    #     allow_unused=True,
    #     materialize_grads=True
    # )[0]
    #
    # u_y = torch.squeeze(u_y, dim=1)
    # u_y = torch.unsqueeze(u_y, dim=1)
    # print("u_y shape", u_y.shape)
    #
    # u_y_new = torch.clone(u_y)
    #
    # u_yy = torch.autograd.grad(
    #     u_y_new, torch.clone(inputs[:,1]).unsqueeze(1),
    #     grad_outputs=torch.ones_like(u),
    #     retain_graph=True,
    #     allow_unused=True,
    #     create_graph=True
    # )[0]


    true_f = f_true(x,y)



    # print("sigma is ", sigma)

    # Extracting individual gradients from the tuple
    # grad_u_x, grad_u_y = torch.autograd.grad(
    #     u, [x, y],
    #     grad_outputs=torch.ones_like(u),
    #     retain_graph=True,
    #     create_graph=True
    # )

    # Calculating the interior loss using the gradients
    # interior_loss = torch.norm(
    #     (torch.squeeze(sigma) - (torch.squeeze(grad_u_x) + torch.squeeze(grad_u_y)))) ** 2 + torch.norm(
    #     (torch.squeeze(div_sigma) + torch.squeeze(f_true(x, y))))

    # print("In interior loss func")
    # print("sigma.shape: ", sigma.shape)
    # print("torch.squeeze(u_x) + torch.squeeze(u_y)) shape: ", (torch.squeeze(u_x) + torch.squeeze(u_y)).shape)
    # print("torch squeeze div_sigma shape: ", torch.squeeze(div_sigma).shape)
    # print("f_true(x,y) shape", torch.squeeze(f_true(x,y)).shape)
    # print("Whole first term squared shape: ",(torch.squeeze(sigma) - (torch.squeeze(u_x) + torch.squeeze(u_y))).shape)
    # print("_________________________________")


    #12/19 6:30 pm : Changed the +torch.squeeze(f_true(x,y) to -torch.squeeze(f_true(x,y) because I think that's how the calculus should work out
    # return torch.norm((torch.squeeze(sigma) - (torch.squeeze(grad_u_x) + torch.squeeze(grad_u_y))))**2 + torch.norm((-torch.squeeze(div_sigma) - torch.squeeze(f_true(x,y))))
    # print("In interior loss fuc: Interior loss = ",torch.norm(-torch.squeeze(u_xx)-torch.squeeze(u_yy)-torch.squeeze(f_true(x,y))))

    # print("-torch.squeeze(u_xx) shape", -torch.squeeze(u_xx).shape)

    # print("-torch.squeeze(u_xx)-torch.squeeze(u_yy) shape: ", (-torch.squeeze(u_xx)-torch.squeeze(u_yy)).shape)

    # breakpoint()

    return torch.norm(-torch.squeeze(u_xx_clone)-torch.squeeze(u_yy_clone)-torch.squeeze(f_true(x,y)))

    # print("The shape of this object is ",(torch.autograd.grad(u, [x, y], grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)).shape)
    # print("the problem term is ", (torch.autograd.grad(u, [x, y], grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)))

    # interior_loss = torch.norm((torch.squeeze(sigma) - (torch.squeeze(grad_u_x) + torch.squeeze(grad_u_y))))**2 + torch.norm((torch.squeeze(div_sigma) + torch.squeeze(f_true(x, y))))
    #
    # return interior_loss
    # return torch.norm((torch.squeeze(sigma) - torch.squeeze((torch.autograd.grad(u, [x, y], grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True))))) ** 2 + torch.norm(
    #     (torch.squeeze(div_sigma) + torch.squeeze(f_true(x, y))))