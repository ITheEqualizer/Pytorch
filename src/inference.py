"""
Inference script example for PyTorch
"""
import torch
import torch.nn as nn
from train import SimpleModel


def load_model(model_path, device):
    """Load a trained model"""
    model = SimpleModel()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def predict(model, input_data, device):
    """Make predictions"""
    with torch.no_grad():
        input_data = input_data.to(device)
        outputs = model(input_data)
        _, predictions = outputs.max(1)
    return predictions


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = '../models/best_model.pth'
    try:
        model = load_model(model_path, device)
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Model not found at {model_path}. Please train a model first.")
        return
    
    # Create dummy input (replace with your actual input)
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # Make prediction
    predictions = predict(model, dummy_input, device)
    print(f"Prediction: {predictions.item()}")


if __name__ == '__main__':
    main()
