# Multimodal Sentiment Analysis with Compositional Embeddings

This project implements a hybrid model for sentiment classification using textual and audio embeddings.
It leverages compositional distributional semantics where phrases are built from pre-trained adjective-noun embeddings.

### Features
- Uses BERT embeddings for text and pretrained OpenL3 or compositional audio embeddings.
- Combines audio and text embeddings for sentiment classification.
- Trains a hybrid model to classify sentiment on the SST-5 dataset.

### Files
- `dataset.py`: Loads the dataset and audio embeddings
- `model.py`: Defines the hybrid audio-text model
- `train.py`: Trains and evaluates the model
- `requirements.txt`: Python dependencies
- `README.md`: This file
- `.gitignore`: Files and folders to be ignored by Git

### Setup
```bash
pip install -r requirements.txt
```

### Usage
```bash
python train.py
```

Ensure that the required CSV files (e.g., `sst5-raw_*`) are placed in the same folder before running the code.

### Citation
If you use this code, please cite:

- How Does an Adjective Sound Like? Exploring Audio Phrase Composition with Textual Embeddings (CLASP 2024)
