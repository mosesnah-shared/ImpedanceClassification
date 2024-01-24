
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch 			  import nn
from torch.utils.data import Dataset, DataLoader

class Autoencoder( nn.Module ):
    def __init__( self ):
        super( Autoencoder, self ).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(   8, 256 ),
            nn.ReLU( True ),
            nn.Linear( 256,  64 ),
            nn.ReLU( True ),
            nn.Linear(  64,  32 ),
            nn.ReLU( True ),
            nn.Linear(  32,  3 ) )
        # Latent space representation        
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(   3,  32 ),
            nn.ReLU( True ),
            nn.Linear(  32,  64 ),
            nn.ReLU( True ),
            nn.Linear(  64, 256 ),
            nn.ReLU( True ),
            nn.Linear( 256,   8 ),
            nn.Sigmoid( )  )
        # Using sigmoid for the last layer if the input data is normalized between 0 and 1

    def forward( self, x ):
        x = self.encoder( x )
        x = self.decoder( x )
        return x
    
    def encode(self, x):
        return self.encoder(x)    



def data_to_dict( file_path ):
    """
    Reads a text file where: 
        (1) The first line contains keys and the subsequent lines contain data.
        (2) The first 8 elements in each line are treated as floats, and the last two are treated as booleans.
    
    :param file_path: The path to the text file to be read.
    :return: A list of dictionaries, each containing the data from one line of the file.
    """
    with open( file_path, 'r' ) as file:
        
        # Read the first line to get the keys for the dictionaries
        keys = file.readline( ).strip( ).split( )
        
        if len( keys ) != 10:
            print("Invalid key format. There must be 10 keys.")
            return [ ]

        # Initialize a dictionary with each key mapping to an empty list
        data_dict = { key: [] for key in keys }
        
        for line_number, line in enumerate( file, start = 2 ):
            
            # Split line into parts
            parts = line.strip().split()  
            
            if len(parts) != 10:
                print( f"Invalid data format on line { line_number }." )
                continue
            
            # Convert the first 8 elements to float and the last two to boolean
            try:
                data = [ float( part ) if idx < 8 else ( part.lower( ) == 'true') for idx, part in enumerate( parts ) ]
                
            except ValueError as e:
                print( f"Conversion error on line {line_number}: {e}" )
                continue

            # Append each value to its corresponding list in the dictionary
            for key, value in zip( keys, data ):
                data_dict[ key ].append( value )
    
    return data_dict


class CustomDataset( Dataset ):
    def __init__( self, data_dict ):
        # Assuming all lists in data_dict are of the same length and correspond to the features of the data points
        # Transpose to get a shape of (num_samples, num_features)
        self.data = np.array( [ data_dict[ key ] for key in sorted( data_dict.keys( ) ) ] ).T  

    def __len__( self ):
        return len( self.data )

    def __getitem__( self, idx ):
        # Assuming that the data is already scaled/normalized appropriately
        return torch.tensor( self.data[ idx ], dtype = torch.float)


if __name__ == "__main__":
    
    # For the first example, read the square data
    file_dir = "./data/set1/parameters_square.txt"
    
    data = data_to_dict( file_dir )

    # Desired keys
    desired_keys = ['k_x', 'k_y', 'k_z', 'k_A', 'k_B', 'k_C', 'damp_t', 'damp_r']

    # Extracting only the desired keys and their corresponding values
    data_x = { key: data[ key ] for key in desired_keys if key in data }

    # Extracting only success trials
    is_succ = data[ 'success' ]

    # Extracting the dataset
    data_x = { key: data[ key ] for key in desired_keys if key in data }

    # Extracting the dataset which only resulted in success
    data_x_succ = {} 

    for key in desired_keys:
        data_x_succ[ key ] = [ data for data, flag in zip( data_x[ key ], is_succ ) if flag ] 

    print( len( data_x[ 'k_x' ] ), len( data_x_succ[ 'k_x' ] ) )

    # Also, normalize the data from 0 to 1
    # Assuming you have the min and max values for each key
    min_values = {'k_x':   50, 'k_y':   50, 'k_z':   50, 'k_A':   5, 'k_B':   5, 'k_C':   5, 'damp_t': 0.3, 'damp_r': 0.3 }
    max_values = {'k_x': 1000, 'k_y': 1000, 'k_z': 1000, 'k_A': 200, 'k_B': 200, 'k_C': 200, 'damp_t': 1.0, 'damp_r': 1.0 }

    # Normalize the data in filtered_data_dict
    data_x_norm = { key: [(value - min_values[key]) / (max_values[key] - min_values[key]) for value in data_x_succ[ key ] ] for key in data_x }

    # Train the Network
    device = torch.device( "cuda" if torch.cuda.is_available( ) else "cpu" )
    model = Autoencoder( ).to( device )
    
    # Mean Squared Error Loss
    criterion = nn.MSELoss()  
    optimizer = torch.optim.Adam( model.parameters( ), lr = 1e-3, weight_decay = 1e-5 )

    dataset = CustomDataset( data_x_norm )
    data_loader = DataLoader( dataset, batch_size=512, shuffle = True )  

    # Assuming data_loader is your PyTorch DataLoader for your dataset
    num_epochs = 2**14
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
            latent = model.encode(img).cpu().numpy()
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
    plt.zlabel('Dimension 3')    
    plt.grid(True)
    plt.show()    

    # Save the trained model
    torch.save(model.state_dict(), 'autoencode_square.pth')
