# MiniCLIP

MiniCLIP is a minimal contrastive vision-language model prototype designed to align image features from a ViT-based encoder with text embeddings from a transformer-based text encoder.

## Overview
- **Vision Encoder:** Uses a Vision Transformer (ViT) to extract image features.
- **Text Encoder:** Uses a transformer-based model (e.g., GPT‑2) to extract text embeddings.
- **Contrastive Learning:** Trains the model by bringing related image-text pairs closer and pushing unrelated pairs apart.

## Repository Structure
- **data/**: Datasets and data processing scripts.
- **models/**: Model architectures including encoders and the projection module.
- **src/**: Core training and evaluation scripts.
- **experiments/**: Experiment configurations and logs.
- **tests/**: Unit and integration tests.
- **docs/**: Documentation and design notes.

## Getting Started

1. Clone the repository.
 ```cmd
    git clone https://github.com/sabrikhalil/MiniCLIP.git
    cd MiniCLIP.
 ```
2. Install dependencies from `requirements.txt`.
 ```cmd
    pip install -r requirements.txt
```
3. Follow the instructions in `docs/design.md` for further details.
 ```cmd
    python data/download_dataset.py
```
4. Training :
 ```cmd
    python src/training.py
```
5. Evaluation: 
 ```cmd
    python src/evaluate.py
```
