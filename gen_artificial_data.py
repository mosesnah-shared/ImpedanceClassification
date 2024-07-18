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
def generate_data( ):
    # The min/max values of   k_x	  k_y,	  k_z,	 k_A,   k_B,   k_C, damp_t,	 damp_r
    min_vals = np.array( [   50.0,   50.0,   50.0,   5.0,   5.0,   5.0,    0.1,    0.1 ] )
    max_vals = np.array( [ 1000.0, 1000.0, 1000.0, 200.0, 200.0, 200.0,    1.0,    1.0 ] )

    input_value = np.random.uniform( min_vals, max_vals )
    
    return input_value

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
            file.write( "\t".join( row ) + "\n" )

    print( f"Results written to {filename}" )

if __name__ == "__main__":
    try:
        N = int( input("Enter the number of data points to generate and predict (positive integer): "))
        if N <= 0:
            raise ValueError( "The number of data points must be a positive integer." )
        
    except ValueError as e:
        print( e )

    else:
        # [Note] [Moses C. Nah] [2024.05.17]
        #   We use the Neural Network that is trained without cross validation
        model_path = "models/set2/binary_classification_cylinder_nn_no_cv.pth"  
        model = load_model(model_path, input_size=8)

        # Generate data and predict outputs until N 'success' outputs are obtained
        whole_data = [ ]

        cnt = 1

        while cnt < N+1:

            data = generate_data( )
            predictions = predict( model, data )

            # If over 0.5, predict success!
            float_strings = [ f"{val:.2f}" for val in data ]

            if predictions >= 0.5:
                whole_data.append( float_strings + [ "success" ] )
                cnt += 1

                if( cnt % 100 == 0 ):
                    print( "[Success Data]:" , cnt , " [Whole Dataset]: ", len( whole_data ) )

            else:
                whole_data.append( float_strings + [ "failure" ] )


        # Save results to file
        save_results(whole_data, "./created_data/set2/predicted_data_cylinder.txt" )