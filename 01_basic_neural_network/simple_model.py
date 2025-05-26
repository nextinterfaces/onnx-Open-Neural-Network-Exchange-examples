import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def main():
    # Create an instance of the model
    model = SimpleNet()
    model.eval()  # Set to evaluation mode

    # Create a dummy input
    dummy_input = torch.randn(1, 2)
    
    # Export the model to ONNX
    torch.onnx.export(
        model,                     # PyTorch model
        dummy_input,              # Model input
        "simple_model.onnx",      # Output file name
        export_params=True,       # Store the trained weights
        opset_version=11,         # ONNX version
        input_names=['input'],    # Model's input names
        output_names=['output'],  # Model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    # Load and verify the exported model
    onnx_model = onnx.load("simple_model.onnx")
    onnx.checker.check_model(onnx_model)
    print("ONNX model was successfully created and verified!")

    # Create an ONNX Runtime session and run inference
    ort_session = onnxruntime.InferenceSession("simple_model.onnx")

    # Prepare input for inference
    input_name = ort_session.get_inputs()[0].name
    test_input = np.random.randn(1, 2).astype(np.float32)

    # Run inference
    output = ort_session.run(None, {input_name: test_input})
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Model output shape: {output[0].shape}")
    print(f"Model output: {output[0]}")

if __name__ == "__main__":
    main() 