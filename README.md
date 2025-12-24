# MNIST Neural Network: Implementation from Scratch

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/Library-NumPy-orange.svg)](https://numpy.org/)
[![Status](https://img.shields.io/badge/Status-Academic_Project-green.svg)]()

## 📌 Project Overview
This repository contains a low-level implementation of a Multi-Layer Perceptron (MLP) to classify handwritten digits (MNIST dataset). 

Unlike typical projects that rely on high-level frameworks like Keras or PyTorch for training, **this project implements the core mathematics of Neural Networks from scratch using NumPy**. This includes manual implementation of:
* Forward Propagation
* Backpropagation (Chain Rule)
* Gradient Descent Optimization
* Activation Functions (Sigmoid)

## 🚀 Key Features
* **Zero-Framework Training:** Keras is used *only* for loading the dataset. The entire training logic is built with raw mathematics.
* **Matrix Operations:** Efficient vectorized implementation of weight updates using NumPy.
* **Architecture:**
    * **Input Layer:** 784 neurons (28x28 flattened images).
    * **Hidden Layer:** 10 neurons (Sigmoid activation).
    * **Output Layer:** 10 neurons (Sigmoid activation).

## 🛠️ Mathematical Implementation
The code manually computes the gradients for weight updates. The core learning loop follows these steps:

1.  **Forward Pass:**
    $$H = \sigma(X \cdot W_1)$$
    $$O = \sigma(H \cdot W_2)$$
2.  **Error Calculation:**
    $$E = Y_{target} - O$$
3.  **Backpropagation:**
    Updating weights based on the derivative of the sigmoid function and the chain rule to propagate the error back from Output to Hidden layer.

## 📊 Results
* **Dataset:** MNIST (subset used for training efficiency in this demo).
* **Accuracy:** Achieved **~80% accuracy** on the test set with this simple architecture.
* **Sample Output:**
    ```text
    Target: [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.] (Digit 7)
    Prediction: [0.07, 0.11, ..., 0.25, ...] (Highest probability at index 7)
    ```

## 💻 Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/hediehmr/MNIST-From-Scratch.git](https://github.com/hediehmr/MNIST-From-Scratch.git)
    cd MNIST-From-Scratch
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy tensorflow keras
    ```
    *(Note: TensorFlow/Keras is required only for downloading the MNIST dataset).*

3.  **Run the training:**
    ```bash
    python src/HandwrittenDigits.py
    ```

## 📧 Contact
**Hedieh Moftakhari Rostamkhani** ML Systems Engineer | [hedieh.rm@gmail.com](mailto:hedieh.rm@gmail.com)
