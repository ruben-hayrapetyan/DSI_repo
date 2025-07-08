import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import re
import string
from collections import Counter
import gc

# === Load datasets ===
imdb = load_dataset("imdb")
cola = {
    "train": load_dataset("shivkumarganesh/CoLA", split="train[:6840]"),
    "test": load_dataset("shivkumarganesh/CoLA", split="train[6840:]")
}
# === Preprocessing ===
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === Tokenization & Vocab ===
def tokenize(text):
    return clean_text(text).split()

# Build vocab from IMDB
counter = Counter()
for example in imdb["train"]:
    counter.update(tokenize(example["text"]))

# Only keep words with min_freq ≥ 5
min_freq = 5
vocab = {"<pad>": 0, "<unk>": 1}
for word, freq in counter.items():
    if freq >= min_freq:
        vocab[word] = len(vocab)
inv_vocab = {v: k for k, v in vocab.items()}
PAD_IDX = vocab["<pad>"]
UNK_IDX = vocab["<unk>"]

def text_pipeline(text, max_len=256):
    tokens = tokenize(text)
    ids = [vocab.get(token, UNK_IDX) for token in tokens[:max_len]]
    return torch.tensor(ids, dtype=torch.long)

# === Dataset Wrappers ===
class TextDataset(Dataset):
    def __init__(self, hf_dataset, label_transform, max_len=256):
        self.data = hf_dataset
        self.label_transform = label_transform
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text_field = item["text"] if "text" in item else item["content"]
        tokens = text_pipeline(text_field, max_len=self.max_len)
        label = torch.tensor(self.label_transform(item["label"]), dtype=torch.long)
        return tokens, label

def collate_batch(batch):
    texts, labels = zip(*batch)
    padded = pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)
    return padded, torch.stack(labels)

# === Dataloaders ===
train_loader_imdb = DataLoader(TextDataset(imdb["train"], lambda x: x), batch_size=64, shuffle=True, collate_fn=collate_batch)
train_loader_cola = DataLoader(TextDataset(cola["train"], lambda x: x), batch_size=256, shuffle=True, collate_fn=collate_batch)
test_loader_cola = DataLoader(TextDataset(cola["test"], lambda x: x), batch_size=256, shuffle=False, collate_fn=collate_batch)

# === Model ===
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextClassifier(len(vocab), num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# === Train on IMDB ===
print("Training on IMDB...")
for epoch in range(10):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in train_loader_imdb:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    print(f"[Epoch {epoch+1}/10] Loss: {total_loss/total:.4f}, Accuracy: {100*correct/total:.2f}%")
    gc.collect()

# === Adapt to cola ===
print("Adapting model for cola...")
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# === Train on cola ===
for epoch in range(3):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in train_loader_cola:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    print(f"[Epoch {epoch+1}/3] Loss: {total_loss/total:.4f}, Accuracy: {100*correct/total:.2f}%")
    gc.collect()

# === Test ===
model.eval()
total_loss, correct, total = 0.0, 0, 0
with torch.no_grad():
    for x, y in test_loader_cola:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
print(f"\nTest Loss: {total_loss/total:.4f}, Accuracy: {100*correct/total:.2f}%")