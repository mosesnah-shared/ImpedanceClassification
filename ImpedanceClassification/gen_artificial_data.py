# Import pip Libraries
import torch
import torch.nn as nn
import numpy    as np
from sklearn.preprocessing import MinMaxScaler

# Import local Library, the Neural Network
from networks import BinaryClassificationNN

# [WARNING] [Moses C. Nah] [2024.05.17]
    #   Care is required when we import the Neural Network, as the 
    #   Neural Network Architecture of the trained and imported models must be identical.
    #   Currently, we used the Neural Network Architecture of 8x64x64x1.
def load_model( model_path, input_size ):
    model = BinaryClassificationNN( input_size = input_size )
    model.load_state_dict( torch.load( model_path ) )
    model.eval( )
    return model

# Since the Neural Network was trained with normalized data range from 0 to 1, 
# Normalize the data with min/max scale
def normalize_input( input_data ):

    # The min/max values of   k_x	  k_y,	  k_z,	 k_A,   k_B,   k_C, damp_t,	 damp_r
    min_vals = np.array( [   50.0,   50.0,   50.0,   5.0,   5.0,   5.0,    0.1,    0.1 ] )
    max_vals = np.array( [ 1000.0, 1000.0, 1000.0, 200.0, 200.0, 200.0,    1.0,    1.0 ] )

    # Scaling with min/max value 
    scaler = MinMaxScaler()
    scaler.fit( [ min_vals, max_vals ] )
    normalized_data = scaler.transform( [ input_data ] )

    return normalized_data

# Generating random data
def generate_data( N ):
    # The min/max values of   k_x	  k_y,	  k_z,	 k_A,   k_B,   k_C, damp_t,	 damp_r
    min_vals = np.array( [   50.0,   50.0,   50.0,   5.0,   5.0,   5.0,    0.1,    0.1 ] )
    max_vals = np.array( [ 1000.0, 1000.0, 1000.0, 200.0, 200.0, 200.0,    1.0,    1.0 ] )

    data = [ ]
    
    for _ in range( N ):
        input_values = np.random.uniform( min_vals, max_vals )
        data.append( list( input_values ) )
    
    return data

# Predicting the output by Binary Classier Neural Network
def predict( model, input_data ):

    # If valid input, normalize from 0 to 1 scale
    normalized_data = normalize_input( input_data )

    # Tensorize the input data, which must be done to avoid annoying warning messages
    input_tensor = torch.tensor( np.array( [ normalized_data ] ), dtype = torch.float32 )

    with torch.no_grad():
        output = model( input_tensor )
    return output.item( )

# Saving the results as txt file
def save_results( data, filename ):
    with open( filename, "w" ) as file:

        # Tacitly, features 1 to 8 are:
        # The min/max values of   k_x	  k_y,	  k_z,	 k_A,   k_B,   k_C, damp_t,	 damp_r
        header = "\t".join( [ f"feature_{ i+1 }" for i in range( 8 ) ] + [ "predicted_output" ] ) + "\n" 
        file.write( header )

        for row in data:
            formatted_row = [ f"{x:.2f}" if isinstance( x, float ) else str( x ) for x in row ]
            file.write( "\t".join( formatted_row ) + "\n" )

    print( f"Results written to {filename}" )

if __name__ == "__main__":
    try:
        N = int( input("Enter the number of 'success' data points to generate and predict (positive integer): "))
        if N <= 0:
            raise ValueError( "The number of data points must be a positive integer." )
        
    except ValueError as e:
        print( e )

    else:
        # [Note] [Moses C. Nah] [2024.05.17]
        #   We use the Neural Network that is trained without cross validation
        model_path = "models/binary_classification_nn_no_cv1.pth"  
        model = load_model(model_path, input_size=8)

        # Generate data and predict outputs until N 'success' outputs are obtained
        success_data = [ ]

        while len( success_data ) < N:
            # [Note] [Moses C. Nah] [2024.05.17]
            #   Since we only use 1 as the argument for generate_data, the function generate_data
            #   seems redundant. However, this function will be later used when we want to generate
            #   N random datasets including both success AND failure.
            #   Currently, we are only interested in having success data, hence just use 1 as argument

            data = generate_data( 1 )[0]
            predictions = predict( model, data )

            # If over 0.5, predict success!
            if predictions >= 0.5:
                success_data.append( data + [ "success" ] )

        # Save results to file
        save_results(success_data, "./created_data/predicted_success_data.txt" )
