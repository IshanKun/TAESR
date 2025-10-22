# TRM-Based Adaptive Embedding Scalable Retrieval System (TAESR)

The TRM-Based Adaptive Embedding Scalable Retrieval System (TAESR) is a demo version of a machine learning-based system that leverages advanced architectures for scalable retrieval. The system is built using PyTorch and integrates recursive deep learning models with attention mechanisms, MLP mixers, and a tokenizer for efficient and adaptive document retrieval.

## Features

- **Tokenization & Embedding**: Uses a custom tokenizer that builds a vocabulary from text input, encoding and decoding text sequences into token IDs.
- **Transformer-based Model**: Implements a recursive transformer model using a combination of MLP mixers and self-attention layers.
- **Adaptive Embeddings**: The system adjusts its embedding depth based on the complexity of the input text, optimizing performance and resource usage.
- **Cache Management**: Persistent in-memory and disk-based caches optimize retrieval times.
- **Recursive Reasoning**: Implements recursive refinement steps for reasoning over input sequences, with an early exit mechanism when convergence criteria are met.
- **Training & Inference**: Supports training with AdamW optimizer, mixed precision, and gradient checkpointing. Includes validation, training checkpointing, and model versioning.
- **Production-Ready**: Embedding caching, live model updates, and drift detection for production environments.

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.10+ (with CUDA support for GPU acceleration)
- Other dependencies:
  - numpy
  - pickle
  - json
  - torch

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/IshanKun/TAESR.git
    cd {dir}
    ```

2. Install required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Download pre-trained models or train a model from scratch (check the training section below).

## Configuration

The core configuration of the system is handled through the `TRMConfig` class. Configuration settings can be saved and loaded in JSON format.

- **architecture**: Choose between `mlp_mixer`, `self_attention`, or `hybrid` models.
- **hidden_dim**: Number of hidden units in the model.
- **max_seq_length**: Maximum sequence length for inputs.
- **vocab_size**: Size of the vocabulary.
- **batch_size**: Batch size for training.
- **max_epochs**: Number of epochs for training.
- **learning_rate**: Learning rate for the optimizer.
