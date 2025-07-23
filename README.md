# Neural Network in C

<p align="center">
  <img src="https://user-images.githubusercontent.com/7684140/27337013-2e2e6e7c-55b2-11e7-9e3a-2b7b6b2c7b2a.png" alt="Neural Network Banner" width="600"/>
</p>

<p align="center">
  <b>A simple yet powerful neural network implementation in C</b><br>
  <i>Learn, experiment, and understand the math behind neural networks!</i>
</p>

---

## 🚀 Features

<ul>
  <li><b>Written in C</b>: Fast and lightweight</li>
  <li><b>Customizable Architecture</b>: Easily modify layers and neurons</li>
  <li><b>CSV Data Support</b>: Train and test with your own datasets</li>
  <li><b>Educational</b>: Perfect for learning neural network basics</li>
</ul>

---

## 📂 Project Structure

```text
├── Nn.c           # Main neural network implementation
├── x_train.csv    # Training features
├── y_train.csv    # Training labels
├── x_test.csv     # Test features
├── y_test.csv     # Test labels
└── README.md      # Project documentation
```

---

## 🛠️ Getting Started

### Prerequisites

- GCC or any C compiler
- Make (optional)

### Build & Run

```bash
# Compile
gcc Nn.c -o nn -lm

# Run
./nn
```

---

## 📊 Dataset Format

- <b>x_train.csv / x_test.csv</b>: Features (each row = sample)
- <b>y_train.csv / y_test.csv</b>: Labels (each row = label)

---

## 🧠 How It Works

<ol>
  <li><b>Initialize Network</b>: Define layers and neurons</li>
  <li><b>Forward Propagation</b>: Calculate outputs</li>
  <li><b>Backpropagation</b>: Adjust weights using error</li>
  <li><b>Training</b>: Iterate over dataset to minimize loss</li>
</ol>

---

## 🧮 Key Equations

<details>
<summary><b>Activation Functions</b></summary>

<br>
<b>Sigmoid:</b>

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

<b>ReLU:</b>

$$
\text{ReLU}(z) = \max(0, z)
$$
</details>

<details>
<summary><b>Weight Initialization</b></summary>

<br>
<b>Xavier Initialization (Output Layer):</b>

$$
W \sim \mathcal{N}\left(0, \frac{1}{n_{in}}\right)
$$

<b>He Initialization (Hidden Layers):</b>

$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)
$$
</details>

<details>
<summary><b>Loss Function</b></summary>

<br>
<b>Binary Cross-Entropy:</b>

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \left[y_i \log(y_i^\text{hat}) + (1 - y_i) \log(1 - y_i^\text{hat})\right]
$$
</details>

<details>
<summary><b>Forward Propagation</b></summary>

<br>
For each layer $l$:

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
$$
$$
a^{(l)} = \text{activation}(z^{(l)})
$$
</details>

<details>
<summary><b>Backpropagation</b></summary>

<br>
For output layer:

$$
\delta^{(L)} = a^{(L)} - y
$$

For hidden layers:

$$
\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \text{activation}'(z^{(l)})
$$

Weight and bias updates:

$$
W^{(l)} = W^{(l)} - \eta \frac{\partial L}{\partial W^{(l)}}
$$
$$
b^{(l)} = b^{(l)} - \eta \frac{\partial L}{\partial b^{(l)}}
$$

Where:
- $\eta$ is the learning rate
- $\odot$ is element-wise multiplication
- $a^{(l)}$ is the activation of layer $l$
- $z^{(l)}$ is the weighted input to layer $l$
- $\delta^{(l)}$ is the error term for layer $l$
</details>

---

## ✨ Example Usage

```c
// ...existing code...
// Example: Training the network
train_network(x_train, y_train);
// ...existing code...
```

---

## 📈 Results

After training, the network predicts outputs for test data and prints accuracy.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Awesome C](https://github.com/kozross/awesome-c)

---

## 📬 Contact

<b>Author:</b> Roshan Chhetri  
<b>GitHub:</b> <a href="https://github.com/Roshan-Chhetri28">Roshan-Chhetri28</a>

---

## 🌱 Future Plans

- Support for batch training instead of SGD
- Additional activation functions like Leaky ReLU or Softmax
- Dynamic learning rate adjustments for better convergence
# Neural Network in C

![Neural Network Banner](https://user-images.githubusercontent.com/7684140/27337013-2e2e6e7c-55b2-11e7-9e3a-2b7b6b2c7b2a.png)

A simple yet powerful implementation of a Neural Network in C. This project demonstrates the fundamentals of neural networks, including forward propagation, backpropagation, and training using CSV datasets.

---

## 🚀 Features
- **Written in C**: Fast and lightweight
- **Customizable Architecture**: Easily modify layers and neurons
- **CSV Data Support**: Train and test with your own datasets
- **Educational**: Perfect for learning neural network basics

---

## 📂 Project Structure
```
├── Nn.c           # Main neural network implementation
├── x_train.csv    # Training features
├── y_train.csv    # Training labels
├── x_test.csv     # Test features
├── y_test.csv     # Test labels
└── README.md      # Project documentation
```

---

## 🛠️ Getting Started

### Prerequisites
- GCC or any C compiler
- Make (optional)

### Build & Run
```bash
# Compile
gcc Nn.c -o nn -lm

# Run
./nn
```

---

## 📊 Dataset Format
- **x_train.csv / x_test.csv**: Features (each row = sample)
- **y_train.csv / y_test.csv**: Labels (each row = label)

---

## 🧠 How It Works
1. **Initialize Network**: Define layers and neurons
2. **Forward Propagation**: Calculate outputs
3. **Backpropagation**: Adjust weights using error
4. **Training**: Iterate over dataset to minimize loss

---

## 🧮 Key Equations

### Activation Functions

**Sigmoid:**

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**ReLU:**

$$
\text{ReLU}(z) = \max(0, z)
$$

### Weight Initialization

**Xavier Initialization (Output Layer):**

$$
W \sim \mathcal{N}\left(0, \frac{1}{n_{in}}\right)
$$

**He Initialization (Hidden Layers):**

$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)
$$

### Loss Function

**Binary Cross-Entropy:**

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \left[y_i \log(y_i^\text{hat}) + (1 - y_i) \log(1 - y_i^\text{hat})\right]
$$

### Forward Propagation

For each layer $l$:

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
$$
$$
a^{(l)} = \text{activation}(z^{(l)})
$$

### Backpropagation

For output layer:

$$
\delta^{(L)} = a^{(L)} - y
$$

For hidden layers:

$$
\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \text{activation}'(z^{(l)})
$$

Weight and bias updates:

$$
W^{(l)} = W^{(l)} - \eta \frac{\partial L}{\partial W^{(l)}}
$$
$$
b^{(l)} = b^{(l)} - \eta \frac{\partial L}{\partial b^{(l)}}
$$

Where:
- $\eta$ is the learning rate
- $\odot$ is element-wise multiplication
- $a^{(l)}$ is the activation of layer $l$
- $z^{(l)}$ is the weighted input to layer $l$
- $\delta^{(l)}$ is the error term for layer $l$

---

---

## ✨ Example Usage
```c
// ...existing code...
// Example: Training the network
train_network(x_train, y_train);
// ...existing code...
```

---

## 📈 Results
After training, the network predicts outputs for test data and prints accuracy.

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License
This project is licensed under the MIT License.

---

## 🙏 Acknowledgements
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Awesome C](https://github.com/kozross/awesome-c)

---

## 📬 Contact
**Author:** Roshan Chhetri  
**GitHub:** [Roshan-Chhetri28](https://github.com/Roshan-Chhetri28)
