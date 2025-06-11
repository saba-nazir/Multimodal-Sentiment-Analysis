import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

class HybridDataset(Dataset):
    def __init__(self, dataframe, audio_embeddings, emb_dim, max_length=66, binary=False):
        self.dataframe = dataframe
        self.audio_embeddings = audio_embeddings
        self.emb_dim = emb_dim
        self.max_length = max_length
        self.binary = binary

    def _pad_sequence(self, sequence):
        return sequence + [0] * (self.max_length - len(sequence)) if len(sequence) < self.max_length else sequence[:self.max_length]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        text = "[CLS] " + row['text'] + " [SEP]"
        tokenized = tokenizer.encode(text, add_special_tokens=False, max_length=self.max_length, pad_to_max_length=True, truncation=True)
        padded_text = self._pad_sequence(tokenized)

        label = row['label']
        audio_vector = self.audio_embeddings.loc[row['text']].values if row['text'] in self.audio_embeddings.index else np.zeros(self.emb_dim)

        return (
            torch.tensor(padded_text, dtype=torch.long),
            torch.tensor(audio_vector, dtype=torch.float),
            torch.tensor(1 if np.any(audio_vector) else 0, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

def load_audio_embeddings(path):
    df = pd.read_csv(path, header=None)
    df.columns = ['text'] + [f'audio_dim_{i}' for i in range(df.shape[1] - 1)]
    return df.set_index('text')