import torch

# Define the limits of integration
x_lower, x_upper = -1, 1
y_lower, y_upper = -1, 1

# Define the constant function z = 1
def constant_function(x, y):
    return torch.ones_like(x)

# Generate sample points for integration
num_points_x = 100
num_points_y = 100
x = torch.linspace(x_lower, x_upper, num_points_x)
y = torch.linspace(y_lower, y_upper, num_points_y)
X, Y = torch.meshgrid(x, y)

# Calculate the function values at the sample points
Z = constant_function(X, Y)

# Calculate the integral using trapezoidal rule
integral_value = torch.trapz(torch.trapz(Z, x), y)

print("Integral value:", integral_value.item())
