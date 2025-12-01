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
    git clone <repo-url>
    cd <repo-directory>
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

#########################################################################################################################################

## TAESRv2: Package Overview

1. Core Model Implementation (TAESRv2.py)

The heart of TAESR combines a compact Transformer backbone with adaptive routing and multi-dimensional embeddings.

- PyTorch + HuggingFace integration: Ready out of the box.
- TAESRConfig: Validated, production-oriented configuration.
- TAESRModel: Compact 7–20M parameter model.
- FastREMRouter: Predicts complexity (easy / medium / hard).
- RecursiveTRMBlock: Shared-weight refinement across iterations.
- Multi-Dim Embeddings: Semantic, temporal, procedural, contextual.
- Hybrid Retrieval Heads: Dense ANN + SPLADE + ColBERT.
- Matryoshka Representations: 64 → 128 → 256 → 384-dim hierarchical slicing.
- Compression Tools: Quantization, PCA, sparsity for 5–10× reduction.

Result: A single model that adapts its depth, chooses the right search strategy, and scales from mobile devices to large-scale RAG systems.

2. Training Pipeline (Artifact: training_scrpt.py)

A full production training suite optimized for performance, stability, and distributed operation:
- ✔ Distributed (DDP/FSDP)
- ✔ Mixed precision (FP16/BF16)
- ✔ Gradient checkpointing for low-memory training

Additional features:
- Multi-objective loss: Dense + sparse + contrastive signals.
- W&B logging: Fully integrated.
- Checkpoint management: Automatic cleanup.
- Schedulers: Warmup + cooldown implemented.

This pipeline allows TAESR to train efficiently even on consumer GPUs.

3. Evaluation Suite (Artifact: v2_eval_bench.py)

A comprehensive benchmarking framework to validate quality, speed, and scaling behavior:
- ✔ BEIR-compatible evaluation
- Metrics: MRR@10, NDCG@10, Recall@50/100, MAP@100
- Profiling: Latency (p50/p95)
- Ablations: Recursion depth, Matryoshka slicing, router on/off
- Reports: Memory + FLOPs analysis
- Benchmarking: End-to-end retrieval performance

Key Features Implemented

Feature                | Status | Description
-----------------------|-------:|------------------------------------------------------
Adaptive Routing       |   ✔   | FastREM picks 1 / 3 / 6 recursion steps automatically
Recursive Refinement   |   ✔   | Shared weights → efficient deep reasoning
Multi-Dim Embeddings   |   ✔   | 4 interpretable subspaces
Matryoshka Learning    |   ✔   | 64 → 384 dims with no retraining
Hybrid Retrieval       |   ✔   | Dense + SPLADE + ColBERT RRF
Compression            |   ✔   | 5–10× size reduction
Gradient Checkpointing |   ✔   | Train on small GPUs
Distributed Training   |   ✔   | Multi-GPU ready
Production Pipeline    |   ✔   | Caching, batching, FP16 inference

TAESR isn’t just an embedding model — it’s a full retrieval ecosystem.

Quick Start

```python
# Initialize model
from TAESRv2 import TAESRModel, TAESRConfig

config = TAESRConfig(
    hidden_size=384,
    recursion_steps={"easy": 1, "medium": 3, "hard": 6}
)
model = TAESRModel(config)

# Inference with adaptive routing
outputs = model(input_ids, attention_mask=mask, return_dict=True)
embedding = outputs.pooler_output  # [Batch, 384]

# Matryoshka slicing for efficiency
embedding_64 = embedding[:, :64]    # Ultra-fast mobile/edge use
embedding_256 = embedding[:, :256]  # Standard retrieval pipelines
```

Innovation Highlights

Think Fast, Think Deep
- TAESR adjusts compute based on query complexity — from 1 loop for simple inputs to 6 loops for complex reasoning.

One Model, Many Workloads
- Supports retrieval, RAG, ranking, representation learning, and classification without retraining.

Explainable Embeddings
- Multi-dimensional embedding space:
  - Semantic
  - Temporal
  - Procedural
  - Contextual

Small But Mighty
- ~10M params with competitive performance to 100M-class embedding models.

Expected Performance in v2
- Speed: 5–8× faster vs fixed-depth transformers.
- Accuracy: Retains >96% performance after 5× compression.
- Scalability: Works at millions-scale with ANN + caching.
- Cost: Fits in low-memory environments and edge devices.

