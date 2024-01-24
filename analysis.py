
from scipy.io import savemat

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch 			  import nn

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

# Load the entire model
# Instantiate the model
model = Autoencoder()

state_dict = torch.load( 'autoencode_square.pth' ) 

model.load_state_dict( state_dict )
model.eval( )

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

# Extracting the dataset which resulted in success/failure
data_x_succ = {} 
data_x_fail = {}

for key in desired_keys:
    data_x_succ[ key ] = [ data for data, flag in zip( data_x[ key ], is_succ ) if     flag ] 
    data_x_fail[ key ] = [ data for data, flag in zip( data_x[ key ], is_succ ) if not flag ] 

# Also, normalize the data from 0 to 1
# Assuming you have the min and max values for each key
min_values = {'k_x':   50, 'k_y':   50, 'k_z':   50, 'k_A':   5, 'k_B':   5, 'k_C':   5, 'damp_t': 0.3, 'damp_r': 0.3 }
max_values = {'k_x': 1000, 'k_y': 1000, 'k_z': 1000, 'k_A': 200, 'k_B': 200, 'k_C': 200, 'damp_t': 1.0, 'damp_r': 1.0 }

# Normalize the data in filtered_data_dict
data_x_succ_norm = { key: [(value - min_values[key]) / (max_values[key] - min_values[key]) for value in data_x_succ[ key ] ] for key in data_x_succ }
data_x_fail_norm = { key: [(value - min_values[key]) / (max_values[key] - min_values[key]) for value in data_x_fail[ key ] ] for key in data_x_fail }

tmp1 = [ torch.tensor( value ) for value in data_x_succ_norm.values( ) ]
tmp2 = [ torch.tensor( value ) for value in data_x_fail_norm.values( ) ]

input_tensor1 = torch.stack( tmp1, dim=0 )
input_tensor2 = torch.stack( tmp2, dim=0 )


with torch.no_grad( ):
    val_succ = model.encode( input_tensor1.T)
    val_fail = model.encode( input_tensor2.T )

latent_succs  = val_succ.cpu( ).numpy( )
latent_fails  = val_fail.cpu( ).numpy( )

# Assuming the latent space is 2-dimensional
xs = latent_succs[:, 0]  # First dimension
ys = latent_succs[:, 1]  # Second dimension
zs = latent_succs[:, 2]  # Third dimension


# Assuming the latent space is 2-dimensional
xf = latent_fails[:, 0]  # First dimension
yf = latent_fails[:, 1]  # Second dimension
zf = latent_fails[:, 2]  # Third dimension

fig = plt.figure(figsize=(8, 6))
ax = plt.axes( projection='3d' )
ax.scatter3D( xs, ys, zs, alpha=0.5, marker='o')
ax.scatter3D( xf, yf, zf, alpha=0.5, marker='d')
plt.title('3D Latent Space')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()    

data = { "xs": xs, "ys": ys, "zs": zs, "xf": xf, "yf": yf, "zf": zf}
savemat('tmp.mat', data)

# Save the values as mat file
