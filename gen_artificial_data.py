import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from networks import BinaryClassificationNN

def load_model(model_path, input_size):
    model = BinaryClassificationNN(input_size=input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def normalize_input(input_data):
    min_vals = np.array([50.0, 50.0, 50.0, 5.0, 5.0, 5.0, 0.1, 0.1])
    max_vals = np.array([1000.0, 1000.0, 1000.0, 200.0, 200.0, 200.0, 1.0, 1.0])
    
    scaler = MinMaxScaler()
    scaler.fit([min_vals, max_vals])
    
    normalized_data = scaler.transform(input_data)
    return normalized_data

def generate_data(N):
    min_vals = np.array([50.0, 50.0, 50.0, 5.0, 5.0, 5.0, 0.1, 0.1])
    max_vals = np.array([1000.0, 1000.0, 1000.0, 200.0, 200.0, 200.0, 1.0, 1.0])
    
    data = []
    
    for _ in range(N):
        input_values = np.random.uniform(min_vals, max_vals)
        data.append(list(input_values))
    
    return data

def predict(model, input_data):
    normalized_data = normalize_input(input_data)
    input_tensor = torch.tensor(normalized_data, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor).numpy()
    return output

def save_results(data, filename="./created_data/predicted_success_data.txt"):
    with open(filename, "w") as file:
        header = "\t".join([f"feature_{i+1}" for i in range(8)] + ["predicted_output"]) + "\n"
        file.write(header)
        for row in data:
            formatted_row = [f"{x:.2f}" if isinstance(x, float) else str(x) for x in row]
            file.write("\t".join(formatted_row) + "\n")
    print(f"Results written to {filename}")

if __name__ == "__main__":
    try:
        N = int(input("Enter the number of 'success' data points to generate and predict (positive integer): "))
        if N <= 0:
            raise ValueError("The number of data points must be a positive integer.")
    except ValueError as e:
        print(e)
    else:
        # Load the model
        model_path = "models/binary_classification_nn_no_cv1.pth"  # or binary_classification_nn_no_cv.pth based on your need
        model = load_model(model_path, input_size=8)

        # Generate data and predict outputs until N 'success' outputs are obtained
        success_data = []

        while len(success_data) < N:
            data = generate_data(1)
            predictions = predict(model, data)
            if predictions[0] >= 0.5:
                success_data.append(data[0] + ["success"])

        # Save results to file
        save_results(success_data)
