import torch
import numpy    as np
from networks import BinaryClassificationNN
from sklearn.preprocessing import MinMaxScaler


def load_model(model_path, input_size):
    model = BinaryClassificationNN(input_size=input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def validate_input(input_data):
    min_vals = np.array([50.0, 50.0, 50.0, 5.0, 5.0, 5.0, 0.1, 0.1])
    max_vals = np.array([1000.0, 1000.0, 1000.0, 200.0, 200.0, 200.0, 1.0, 1.0])

    for i, value in enumerate(input_data):
        if not (min_vals[i] <= value <= max_vals[i]):
            raise ValueError(f"Input value {value} at index {i} is out of the allowed range [{min_vals[i]}, {max_vals[i]}]")

def normalize_input(input_data):
    min_vals = np.array([50.0, 50.0, 50.0, 5.0, 5.0, 5.0, 0.1, 0.1])
    max_vals = np.array([1000.0, 1000.0, 1000.0, 200.0, 200.0, 200.0, 1.0, 1.0])
    
    scaler = MinMaxScaler()
    scaler.fit([min_vals, max_vals])
    
    normalized_data = scaler.transform([input_data])
    return normalized_data[0]

def predict(model, input_data):
    validate_input(input_data)
    normalized_data = normalize_input(input_data)
    input_tensor = torch.tensor(np.array( [normalized_data] ), dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
    return output.item()


if __name__ == "__main__":
    # Path to the saved model
    model_path = "models/binary_classification_nn_cv1.pth"  # or binary_classification_nn_no_cv.pth based on your need

    # Load the model
    model = load_model(model_path, input_size=8)

    # Repeat input and prediction process until interrupted
    try:
        while True:
            # Get input from the user
            input_data = input("Enter 8 numbers separated by space: ")
            input_data = list(map(float, input_data.split()))

            if len(input_data) != 8:
                raise ValueError("You must enter exactly 8 numbers.")

            try:
                prediction = predict(model, input_data)
                result = "Success" if prediction >= 0.5 else "Failure"
                print(f"Prediction: {result}")
            except ValueError as e:
                print(e)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")