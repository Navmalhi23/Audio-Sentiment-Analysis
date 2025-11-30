import torch
import torch.nn as nn
import os

from torch.utils.data import Dataset, DataLoader

from helpers import *
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        """
        embeddings: list of torch.Tensor, each shape (T_i, feature_dim)
        labels: list of int
        """
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        x = self.embeddings[idx]           
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
    

def collate_fn(batch):
    xs = [torch.tensor(item[0], dtype=torch.float32) for item in batch]
    ys = torch.tensor([item[1] for item in batch], dtype=torch.long)
    xs_padded = pad_sequence(xs, batch_first=True) 
    return xs_padded, ys

class AudioRNN(nn.Module):
    def __init__(self, 
                 feature_dim: int, 
                 hidden_dim: int = 128, 
                 num_layers: int = 2,
                 rnn_type: str = "lstm",
                 num_classes: int = 10):
        super().__init__()

        rnn_cls = {
            "lstm": nn.LSTM,
            "gru": nn.GRU,
            "rnn": nn.RNN
        }[rnn_type.lower()]

        self.rnn = rnn_cls(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: (batch, time, feature_dim)
        """
        out, hidden = self.rnn(x)
        last_out = out[:, -1, :] 
        out = self.fc(last_out)
        return out


def train(embeddings, labels, num_classes=8, device="cpu"):
    # Split train/test manually if needed
    print("Training")
    dataset = EmbeddingDataset(embeddings, labels)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    model = AudioRNN(feature_dim=1024, hidden_dim=128, num_layers=2, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 200
    for epoch in range(epochs):
        print(f"epoch: {epoch}")
        model.train()
        running_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(loader):.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)            
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"Accuracy: {correct/total:.4f}")

    return model


files = get_file_matrix()
file_paths = matrix_to_filename(files, RAW_DATA_PATH)
labels = get_labels(files)
embeddings = []
EMBEDDING_FILE = 'embeddings.pt'
print("Getting embeddings")

if(os.path.exists(EMBEDDING_FILE)):
    print(f"Loading embeddings from {EMBEDDING_FILE}")
    embeddings = torch.load(EMBEDDING_FILE)
else:
    print(f"Loading embeddings using Wav2Vec")
    c = 0
    for f in file_paths:
        embeddings.append(torch.from_numpy(get_embedding(f)).unsqueeze(-1).squeeze(0).squeeze(-1))
        c+=1
        print(c)
    torch.save(embeddings, EMBEDDING_FILE)
    print(f"Saved embeddings to {EMBEDDING_FILE}")
print("Done getting embeddings")


train_emb, test_emb, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42
)
print("finished train/test split")
model = train(train_emb, y_train)

correct, total = 0, 0
for emb, label in zip(test_emb, y_test):
    x = emb.unsqueeze(0).to(device) 
    y = torch.tensor([label], device=device)
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
    correct += (pred == y).sum().item()
    total += 1
torch.save(model, "model.pt")
print(f"\nAccuracy: {correct/total:.4f}")
