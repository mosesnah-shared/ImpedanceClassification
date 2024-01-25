import numpy as np
import torch
from torch.utils.data import Dataset

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


