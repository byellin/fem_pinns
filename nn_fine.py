import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#Define neural network that I will train using coarse data samples 
#Use this as an initial guess for the fine scale training
torch.set_default_dtype(torch.float32)
class NeuralNetwork_fine(nn.Module):

    def __init__(self):
        super(NeuralNetwork_fine, self).__init__()

        torch.set_default_dtype(torch.float32)

        self.flatten = nn.Flatten()
        #Input layer is 2 dimensional because I have (x,y) information in data 
        #Output layer is 1 dimensional because I want to output the temperature
        #at that particular point 
        # breakpoint()
        print('about to define linear relu stack')
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2,20),
            nn.Tanh(),
            nn.Linear(20,10),
            nn.Tanh(),
            nn.Linear(10,5),
            nn.Tanh(),
            nn.Linear(5,5),
            nn.Tanh(),
            nn.Linear(5,1)

        )
        print("type of linear relu stack is ", self.linear_relu_stack)
        self.initialize_weights()

    def forward(self, x):
        #x = self.flatten(x)
        x=x.to(torch.float32)
        logits = self.linear_relu_stack(x)
        return logits


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize weights using a normal distribution with mean 0 and standard deviation 0.1
                nn.init.normal_(m.weight, mean=0, std=0.1)
                # Initialize biases to zeros
                nn.init.constant_(m.bias, 0)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model_fine = NeuralNetwork_fine().to(device)
print(model_fine)

#Show data types of model parameters 
for param in model_fine.linear_relu_stack.parameters():
  print(param.data.dtype)

