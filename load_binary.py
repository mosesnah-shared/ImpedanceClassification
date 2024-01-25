import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Define the network architecture
class BinaryClassifier(nn.Module):
	def __init__(self):
		super(BinaryClassifier, self).__init__()
		self.layer_1 = nn.Linear( 8, 256)  # Adjust the input dimension
		self.layer_2 = nn.Linear(256, 32)
		self.layer_out = nn.Linear(32, 1)
		
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, inputs):
		x = self.relu(self.layer_1(inputs))
		x = self.relu(self.layer_2(x))
		x = self.layer_out(x)
		x = self.sigmoid(x)
		return x

# Initialize the model
model = BinaryClassifier()

# Load the model parameters from the file
model_path = 'binary_classifier_model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

# Assuming you have your test data as X_test_tensor
# (you need to preprocess your test data as you did with your training data)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)  # Convert X_test to a PyTorch tensor

# Create a DataLoader for your test data
test_dataset = TensorDataset(X_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Forward pass on the test data
y_pred_list = []
with torch.no_grad():
    for X_batch in test_loader:
        Y_pred = model(X_batch[0])
        y_pred_list.append(Y_pred.numpy())

# Process predictions if necessary (e.g., apply threshold, round)
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

# Now y_pred_list contains the predictions for the test set