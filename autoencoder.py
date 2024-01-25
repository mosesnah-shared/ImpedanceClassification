
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from utils    import data_to_dict, CustomDataset
from networks import Autoencoder

if __name__ == "__main__":
    
    # Types of shapes
    idx    = 1
    shapes = [ "cylinder", "hex", "square", "triangle" ]
    b_size = [        64,   64,     512,      64 ]
    # Read the file and parse
    file_dir = "./data/set1/parameters_" + shapes[ idx ] + ".txt"
    data_raw = data_to_dict( file_dir )

    # Desired keys that we want to extract out
    desired_keys = ['k_x', 'k_y', 'k_z', 'k_A', 'k_B', 'k_C', 'damp_t', 'damp_r']

    # Extracting out the desired keys and their corresponding values
    Xtrain = { key: data_raw[ key ] for key in desired_keys if key in data_raw }

    # Extracting only success trials
    is_succ = data_raw[ 'success' ]

    # Extracting the dataset which only resulted in success
    Xtrain_succ = {} 

    for key in desired_keys:
        Xtrain_succ[ key ] = [ data for data, flag in zip( Xtrain[ key ], is_succ ) if flag ] 

    # Also, normalize the data from 0 to 1
    # Assuming you have the min and max values for each key
    min_values = {'k_x':   50, 'k_y':   50, 'k_z':   50, 'k_A':   5, 'k_B':   5, 'k_C':   5, 'damp_t': 0.3, 'damp_r': 0.3 }
    max_values = {'k_x': 1000, 'k_y': 1000, 'k_z': 1000, 'k_A': 200, 'k_B': 200, 'k_C': 200, 'damp_t': 1.0, 'damp_r': 1.0 }

    # Normalize the data in filtered_data_dict
    Xtrain_succ_norm = { key: [(value - min_values[key]) / (max_values[key] - min_values[key]) for value in Xtrain_succ[ key ] ] for key in Xtrain_succ }

    # Train the Network
    device = torch.device( "cuda" if torch.cuda.is_available( ) else "cpu" )
    model  = Autoencoder( ).to( device )
    
    # Mean Squared Error Loss
    criterion = nn.MSELoss()  
    optimizer = torch.optim.Adam( model.parameters( ), lr = 1e-3, weight_decay = 1e-5 )

    dataset = CustomDataset( Xtrain_succ_norm )

    data_loader = DataLoader( dataset, batch_size=b_size[ idx ], shuffle = True )  

    # Assuming data_loader is your PyTorch DataLoader for your dataset
    num_epochs = 2**12
    losses = np.zeros( num_epochs )  # List to store loss values

    for epoch in range( num_epochs ):
        for batch in data_loader:
            img = batch.to( device )
            
            # Forward pass
            outputs = model( img )
            loss = criterion( outputs, img )
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        losses[ epoch ] = loss.item( ) 

        # Log the loss value every epoch
        print(f'epoch [{epoch+1}/{num_epochs}], loss:{loss.item():.4f}')
    
    # Plot the training losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    # Also plot the latent_space 
    # Assuming data_loader is your DataLoader containing the dataset
    latent_space_representations = []
    for batch in data_loader:
        img = batch.to(device)
        with torch.no_grad():
            latent = model.encode( img ).cpu().numpy()
            latent_space_representations.append(latent)

    # Convert the list of arrays to a single numpy array
    latent_space_representations = np.concatenate(latent_space_representations, axis=0)

    # Assuming the latent space is 2-dimensional
    x = latent_space_representations[:, 0]  # First dimension
    y = latent_space_representations[:, 1]  # Second dimension
    z = latent_space_representations[:, 2]  # Third dimension

    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes( projection='3d' )
    ax.scatter3D(x, y, z, alpha=0.5)
    plt.title('3D Latent Space')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.show()    

    # Save the trained model
    torch.save( model.state_dict(), './models/autoencode_' + shapes[ idx ] + '.pth')
