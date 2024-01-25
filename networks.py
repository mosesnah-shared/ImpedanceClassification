
from torch import nn

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.layer_1 = nn.Linear(8, 256) # First hidden layer
        self.layer_2 = nn.Linear(256, 32) # Second hidden layer
        self.layer_out = nn.Linear(32, 1) # Output layer
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.layer_out(x)
        x = self.sigmoid(x)  # Sigmoid activation for binary classification
        return x

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

