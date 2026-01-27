# Name Generator: Character-Level Language Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/drive/YOUR_NOTEBOOK_LINK_HERE](https://colab.research.google.com/drive/1oqwxt_Uu3YLr6_A-xmggvfqGHCf5zsS0#scrollTo=6vAyxwx6P78O))

A deep learning project that generates unique, sounding names by training on a dataset of 32,000+ names. This repository implements two different architectures from scratch using **PyTorch**, evolving from simple statistical methods to a multi-layer neural network.

Inspired by Andrej Karpathy's "Makemore" series and the research paper *A Neural Probabilistic Language Model* (Bengio et al., 2003).

## 🧠 Architectures Implemented

### 1. Bigram Language Model (`Bigram_Model.ipynb`)
The baseline approach that predicts the next character based solely on the previous one.
*   **Method A (Statistical):** Uses a frequency count matrix (`N`) to calculate probability distributions.
*   **Method B (Neural):** A single-layer neural network (Linear layer + Softmax) that learns the same probability distribution via gradient descent.
*   **Result:** A simple model with high loss (~2.45), often generating gibberish but capturing basic pairs like "qu".

### 2. Multi-Layer Perceptron (MLP) (`MLP_Model.ipynb`)
A more advanced neural network based on Bengio et al. (2003) that uses a larger context window (3 characters) to predict the next one.
*   **Embedding Layer:** Maps characters to a 10-dimensional vector space to capture similarity (e.g., vowels cluster together).
*   **Hidden Layer:** A `Tanh` activation layer with 200 neurons for non-linearity.
*   **Context Window:** Looks at 3 previous characters instead of just 1.
*   **Result:** Significantly lower loss (~2.1) and generates far more realistic names compared to the Bigram model.

## 🛠️ Tech Stack
*   **Python**
*   **PyTorch** (Tensors, Autograd, Neural Net modules)
*   **Matplotlib** (Visualization of embeddings and loss)
*   **Pandas/NumPy** (Data manipulation)

