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
from opacus import PrivacyEngine

# === Load datasets ===
imdb = load_dataset("imdb")

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
    tokens = tokenize(text)[:max_len]
    bow_vector = torch.zeros(len(vocab), dtype=torch.float)
    for token in tokens:
        idx = vocab.get(token, UNK_IDX)
        bow_vector[idx] += 1
    return bow_vector

# === Dataset Wrappers ===
class TextDataset(Dataset):
    def __init__(self, hf_dataset, label_transform, text_field, max_len=256):
        self.data = hf_dataset
        self.label_transform = label_transform
        self.max_len = max_len
        self.text_field = text_field

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = text_pipeline(item[self.text_field], max_len=self.max_len)
        label = torch.tensor(self.label_transform(item["label"]), dtype=torch.long)
        return tokens, label

def collate_batch(batch):
    texts, labels = zip(*batch)
    padded = pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)
    return padded, torch.stack(labels)

# === Dataloaders ===
train_loader_imdb = DataLoader(TextDataset(imdb["train"], lambda x: x, text_field="text"), batch_size=64, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(TextDataset(imdb["test"], lambda x: x, text_field="text"), batch_size=64, shuffle=True, collate_fn=collate_batch)


# === Model ===
class Model(nn.Module):
    def __init__(self, vocab_size, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(vocab_size, num_classes)

    def forward(self, x):
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(len(vocab), num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
privacy_engine = PrivacyEngine()

# === Train on IMDB ===
epochs_2 = 15
train_losses_2 = []
train_accuracies_2 = []
test_losses_2 = []
test_accuracies_2 = []
model.train()
target_delta = 1e-5
model, optimizer, train_loader_private = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader_imdb,
    epochs=epochs_2,
    target_epsilon=3,
    target_delta=target_delta,
    max_grad_norm=0.1,
)
for epoch in range(epochs_2):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader_private:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Epoch [{epoch + 1}/{epochs_2}]')
    print("Training loss", (running_loss / total))
    train_losses_2.append(running_loss / total)
    print("Training accuracy", (correct *100/ total),'%')
    train_accuracies_2.append(correct * 100 / total)
    gc.collect()

    epsilon = privacy_engine.get_epsilon(delta=target_delta)
    print(f"Privacy Budget (epsilon, delta): ({epsilon:.2f}, {target_delta})")

    model.eval()
    running_loss_eval = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss_eval += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Testing loss", (running_loss_eval / total))
    test_losses_2.append(running_loss_eval / total)
    print("Testing accuracy", (correct*100 / total),'%')
    test_accuracies_2.append(correct * 100 / total)

    gc.collect()
    model.eval()
    correct = 0
    total = 0
    eval_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            eval_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

print(f"Test Loss: {eval_loss / total:.4f}, Accuracy: {100 * correct / total:.2f}%")