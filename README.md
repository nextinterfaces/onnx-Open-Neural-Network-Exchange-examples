# ONNX Examples

This repository contains examples demonstrating the usage of ONNX (Open Neural Network Exchange) format and runtime.

## Setup

### Prerequisites
- Python 3.8 or higher
- uv (Python package installer)

### Virtual Environment Setup

1. Create a virtual environment:
```bash
uv venv
```

2. Activate the virtual environment:
```bash
# On macOS/Linux
source .venv/bin/activate

# On Windows
.\.venv\Scripts\activate
```

3. Install dependencies for a specific example:
```bash
cd example_directory  # e.g., cd 01_basic_neural_network
uv pip install -r requirements.txt
```

## Examples

### [01_basic_neural_network](01_basic_neural_network/)
A simple example demonstrating:
- Creating a basic neural network using PyTorch
- Exporting the model to ONNX format
- Loading and running inference using ONNX Runtime

The network architecture includes:
- Input layer (2 neurons)
- Hidden layer (3 neurons) with ReLU activation
- Output layer (1 neuron) with Sigmoid activation

### [02_sentence_similarity](02_sentence_similarity/)
An example using the Redis langcache-embed-v1 model for semantic similarity:
- Loading a pre-trained sentence transformer model
- Computing sentence embeddings and similarities
- Exporting to ONNX format
- Running inference with ONNX Runtime

The model features:
- 768-dimensional embeddings
- 8192 token maximum sequence length
- Fine-tuned on Quora dataset
- High accuracy on semantic similarity tasks

## Contributing

Feel free to contribute additional examples or improvements to existing ones by submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 