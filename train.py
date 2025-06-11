import os
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from src.dataset import HybridDataset, load_audio_embeddings
from src.model import HybridModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

def train_one_epoch(model, loss_fn, optimizer, dataset, batch_size=32):
    model.train()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_loss, total_acc = 0, 0

    for X_text, X_audio, has_audio, labels in tqdm(dataloader, desc="Training"):
        X_text, X_audio, has_audio, labels = X_text.to(device), X_audio.to(device), has_audio.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(X_text, None, X_audio, has_audio)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / len(dataset), total_acc / len(dataset)

def evaluate(model, loss_fn, dataset, batch_size=32):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    total_loss, total_acc = 0, 0

    with torch.no_grad():
        for X_text, X_audio, has_audio, labels in tqdm(dataloader, desc="Evaluating"):
            X_text, X_audio, has_audio, labels = X_text.to(device), X_audio.to(device), has_audio.to(device), labels.to(device)
            logits = model(X_text, None, X_audio, has_audio)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            total_acc += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / len(dataset), total_acc / len(dataset)

if __name__ == "__main__":
    DATA_DIR = "data/"
    train_df = pd.read_csv(os.path.join(DATA_DIR, "sst5-raw_train_label.csv")).rename(columns={0: "text", 1: "label"})
    test_df = pd.read_csv(os.path.join(DATA_DIR, "sst5-raw_test_label.csv")).rename(columns={0: "text", 1: "label"})
    val_df = pd.read_csv(os.path.join(DATA_DIR, "sst5-raw_val_label.csv")).rename(columns={0: "text", 1: "label"})

    train_audio = load_audio_embeddings(os.path.join(DATA_DIR, "sst5-raw_train_audio_embed.csv"))
    test_audio = load_audio_embeddings(os.path.join(DATA_DIR, "sst5-raw_test_audio_embed.csv"))
    val_audio = load_audio_embeddings(os.path.join(DATA_DIR, "sst5-raw_val_audio_embed.csv"))

    emb_dim = len(train_audio.columns)
    model = HybridModel(audio_embedding_dim=emb_dim).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    trainset = HybridDataset(train_df, train_audio, emb_dim)
    valset = HybridDataset(val_df, val_audio, emb_dim)
    testset = HybridDataset(test_df, test_audio, emb_dim)

    for epoch in range(10):
        train_loss, train_acc = train_one_epoch(model, loss_fn, optimizer, trainset)
        val_loss, val_acc = evaluate(model, loss_fn, valset)
        test_loss, test_acc = evaluate(model, loss_fn, testset)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} Acc={train_acc:.4f}, Val Loss={val_loss:.4f} Acc={val_acc:.4f}, Test Loss={test_loss:.4f} Acc={test_acc:.4f}")