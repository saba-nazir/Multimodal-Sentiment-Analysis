# Multimodal Sentiment Analysis with Compositional Embeddings

This project implements a hybrid model for sentiment classification using textual and audio embeddings.
It leverages compositional distributional semantics where phrases are built from pre-trained adjective-noun embeddings.

### Features
- Uses BERT embeddings for text and pretrained OpenL3 or compositional audio embeddings.
- Combines audio and text embeddings for sentiment classification.
- Trains a hybrid model to classify sentiment on the SST-5 dataset.

### Structure
- `src/`: Core model and dataset code
- `scripts/`: Training and evaluation routines
- `data/`: Placeholder for dataset files
- `models/`: For saving trained models

### Setup
```bash
pip install -r requirements.txt
```

### Usage
```bash
python scripts/train.py
```

Ensure that the audio-text embedding CSVs and label files are placed in the `data/` folder before running.

### Citation
If you use this code, please cite:

- [How Does an Adjective Sound Like? Exploring Audio Phrase Composition with Textual Embeddings (CLASP 2024)]
