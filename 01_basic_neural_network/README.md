# Basic ONNX Neural Network Example

This example demonstrates how to:
1. Create a simple neural network using PyTorch
2. Export it to ONNX format
3. Load and run inference using ONNX Runtime

## Model Architecture
The neural network consists of:
- Input layer (2 neurons)
- Hidden layer (3 neurons) with ReLU activation
- Output layer (1 neuron) with Sigmoid activation

## Requirements
Install the required packages:
```bash
uv pip install -r requirements.txt
```

## Running the Example

```bash
uv run simple_model.py
```

This will:
1. Create a simple neural network
2. Export it to ONNX format (`simple_model.onnx`)
3. Verify the ONNX model
4. Run inference with random test data

## Expected Output
The script will print:
1. Confirmation that the ONNX model was created and verified
2. The shape of the test input
3. The shape and value of the model's output