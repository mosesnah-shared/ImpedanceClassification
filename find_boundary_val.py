import torch
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
import scipy.io

from networks import BinaryClassifier
from utils import set_seeds

# Load the trained model
model = BinaryClassifier()
model.load_state_dict(torch.load('./models/binary_classifier_model_square.pth'))
model.eval()  # Set the model to evaluation mode

set_seeds( )
# Function to find the input x that gives an output close to 0.5
def find_boundary_value(model, starting_point, target=0.5, learning_rate=0.01, max_iterations=1000):
    # Initialize x as a parameter with gradient
    x = Variable(torch.tensor(starting_point, dtype=torch.float32), requires_grad=True)
    
    # Use Adam optimizer
    optimizer = Adam([x], lr=learning_rate)
    
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(x)
        
        # Compute loss as the absolute difference from the target
        loss = torch.abs(y_pred - target)
        
        # Backward pass
        loss.backward()
        
        # Update x
        optimizer.step()
        
        # Check if the prediction is close to the target
        if loss.item() < 1e-3:
            break
    
    return x.detach().numpy()

# Function to sample N 8-dimensional arrays that output a value near 0.5
def sample_boundary_values(model, N, dim=8, learning_rate=0.001, max_iterations=1000):
    boundary_values = []
    
    for i in range(N):
        # Generate a random starting point for each sample
        starting_point = np.random.uniform(low=0, high=1, size=dim)
        boundary_x = find_boundary_value(model, starting_point, learning_rate=learning_rate, max_iterations=max_iterations)
        boundary_values.append(boundary_x)

        print(f"Collecting sample {i + 1}/{N}")

    return np.array(boundary_values)

# Sample N 8-dimensional arrays
N = 2**9  # Number of samples you want
sampled_values = sample_boundary_values(model, N)

# Convert sampled_values to PyTorch tensor
sampled_values_tensor = torch.tensor(sampled_values, dtype=torch.float32)

# Pass the tensor through the model
with torch.no_grad():  # We don't need to compute gradients here, so we disable them
    model.eval()  # Set the model to evaluation mode
    outputs = model(sampled_values_tensor)

# Your data
save_data = { 'sampled_vals': sampled_values, 'outputs': outputs.numpy( ) }

# Save data to a .mat file
scipy.io.savemat( './MATLAB_data/data_square_to_sample.mat', save_data)