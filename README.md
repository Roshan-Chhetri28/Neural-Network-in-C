# Neural Network Implementation in C

This C program implements a simple feedforward neural network with backpropagation, supporting ReLU and sigmoid activation functions. It is designed to train on a dataset loaded from CSV files and uses stochastic gradient descent (SGD) for optimization.

## Features
- **Multi-layer Neural Network**: Configurable number of layers and neurons.
- **Activation Functions**:
  - `sigmoid(z)`: Used in the output layer for binary classification.
  - `relu(z)`: Used in hidden layers to introduce non-linearity.
- **Weight Initialization**:
  - **Xavier Initialization** for the output layer.
  - **He Initialization** for hidden layers.
- **Loss Function**:
  - `binary_cross_entropy(y, y_hat)`: Computes loss for binary classification.
- **Training**:
  - Forward pass to compute predictions.
  - Backpropagation to compute gradients.
  - Weight and bias updates using **Stochastic Gradient Descent (SGD)**.

## Key Components

### 1. **Data Structures**
- `Layer`: Represents a single layer in the neural network.
- `NeuralNetwork`: Represents the entire neural network.

### 2. **Forward Pass (`forward_pass`)**
- Propagates input through layers, applying weighted sums and activations.

### 3. **Backward Pass & Weight Updates (`SGD`)**
- Computes gradients using **backpropagation**.
- Updates weights and biases using **learning rate**.

### 4. **Training Function (`train`)**
- Iterates over **epochs**.
- Loads training data, computes loss and accuracy, and updates weights.

### 5. **Loading Data (`load_csv`)**
- Reads CSV files and converts them into a **2D array** of `double` values.

## Execution Flow
1. **Initialize Neural Network** (`createNN`)
2. **Load Training Data** (`load_csv`)
3. **Train the Model** (`train`)
4. **Evaluate Performance** (Loss & Accuracy)

## Example Execution
The program reads `x_train.csv` and `y_train.csv`, initializes a neural network with:
- **3 layers**: Input (2 neurons), Hidden (3 neurons), Output (1 neuron)
- **1000 epochs** with a **learning rate of 0.0001**
- Outputs **loss and accuracy** per epoch.

## Compilation & Execution
```sh
gcc -o neural_net neural_net.c -lm
./neural_net
```
## Future Plans
### Support for batch training instead of SGD.
### Additional activation functions like Leaky ReLU or Softmax.
### Dynamic learning rate adjustments for better convergence.
