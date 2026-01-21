import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import matplotlib.pyplot as plt

class EuropeDataset(Dataset):
    def __init__(self, filename):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        self.data = pd.read_csv(file_path)
        self.features = torch.tensor(self.data[['long', 'lat']].values, dtype=torch.float32)
        self.labels = torch.tensor(self.data['country'].values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MLP(nn.Module):
    def __init__(self, num_hidden_layers, hidden_dim, output_dim):
        super(MLP, self).__init__()
        layers = []
        curr_dim = 2
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.output_layer(self.hidden_layers(x))

def train(train_ds, val_ds, test_ds, model, epochs=30):
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024)
    test_loader = DataLoader(test_ds, batch_size=1024)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_accs = [], []

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x)
                _, pred = torch.max(outputs, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        val_accs.append(correct / total)
        print(f"Epoch {ep+1}: Loss {avg_loss:.4f}, Val Acc {val_accs[-1]:.3f}")

    # --- הוספתי את החלק הזה כאן כדי שיודפס לפני הגרפים ---
    model.eval()
    t_correct, t_total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            _, pred = torch.max(outputs, 1)
            t_correct += (pred == y).sum().item()
            t_total += y.size(0)
    
    print("\n" + "="*30)
    print(f"FINAL TEST ACCURACY: {t_correct/t_total:.4f}")
    print("="*30 + "\n")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.plot(train_losses); plt.title('Loss')
    plt.subplot(1, 2, 2); plt.plot(val_accs); plt.title('Accuracy')
    plt.show()

if __name__ == "__main__":
    train_ds = EuropeDataset('train.csv')
    val_ds = EuropeDataset('validation.csv')
    test_ds = EuropeDataset('test.csv')
    
    num_classes = int(pd.concat([train_ds.data['country'], val_ds.data['country']]).max()) + 1
    model = MLP(3, 64, num_classes)
    
    train(train_ds, val_ds, test_ds, model)