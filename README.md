# Micrograd from Scratch

A lightweight, scalar-valued **Autograd engine** with a PyTorch-like API. This project implements backpropagation (reverse-mode gradient descent) from scratch using only pure Python.

## 🚀 Overview
This engine builds a dynamic computational graph (Directed Acyclic Graph) of mathematical operations. It can compute the derivative of any complex expression by applying the chain rule automatically.


## 🧠 Features
- **Standalone Engine:** No dependency on PyTorch or NumPy for the core math.
- **Dynamic Graph:** Builds the graph on the fly during the forward pass.
- **Neural Network Library:** Includes modules for `Neurons`, `Layers`, and `MLP` (Multi-Layer Perceptron).

## 📊 Convergence Demo
The model successfully converges on a binary classification task. Here is the terminal output from the training loop:

```bash
Iteration [0/30]  | Loss: 4.2339 | Status: Initializing
Iteration [10/30] | Loss: 0.3530 | Status: Learning
Iteration [20/30] | Loss: 0.0975 | Status: Converging
Iteration [29/30] | Loss: 0.0534 | Status: Optimized

🛠️ How to Use

from micrograd import Value, MLP

# Create a neural network (3 inputs, two hidden layers of 4, 1 output)
model = MLP(3, [4, 4, 1])

# Forward pass
inputs = [2.0, 3.0, -1.0]
prediction = model(inputs)

# Backward pass (The magic)
prediction.backward()

# Check gradients of weights
for p in model.parameters():
    print(p.grad)

📂 Project Structure
micrograd.py: The core Autograd engine and Neural Network classes.

.gitignore: Prevents virtual environments and cache files from being uploaded.

Inspired by Andrej Karpathy's micrograd.
