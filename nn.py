import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#Define neural network that I will train using coarse data samples 
#Use this as an initial guess for the fine scale training
torch.set_default_dtype(torch.float32)
class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # torch.set_default_dtype(torch.float64)

        self.flatten = nn.Flatten()
        #Input layer is 2 dimensional because I have (x,y) information in data 
        #Output layer is 1 dimensional because I want to output the temperature
        #at that particular point 
        
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

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

#Show data types of model parameters 
for param in model.linear_relu_stack.parameters():
  print(param.data.dtype)

