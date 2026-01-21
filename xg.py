import os
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from xgboost import XGBClassifier

def get_loaders(path, transform, batch_size):
    # Use absolute path to avoid folder issues
    train_path = os.path.join(path, 'train')
    test_path = os.path.join(path, 'test')
    
    train_set = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    test_set = torchvision.datasets.ImageFolder(root=test_path, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Configuration
np.random.seed(0)
current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_dir, 'whichfaceisreal') 
batch_size = 32
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.CenterCrop(64), 
    transforms.ToTensor()
])

# Load Loaders
train_loader, test_loader = get_loaders(path, transform, batch_size)

# DATA LOADING (The fixed part provided by your instructor)
train_data, train_labels, test_data, test_labels = [], [], [], []
with torch.no_grad():
    for (imgs, labels) in tqdm(train_loader, desc='Processing Train'):
        train_data.append(imgs)
        train_labels.append(labels)
    train_data = torch.cat(train_data, 0).cpu().numpy().reshape(len(train_loader.dataset), -1)
    train_labels = torch.cat(train_labels, 0).cpu().numpy()
    
    for (imgs, labels) in tqdm(test_loader, desc='Processing Test'):
        test_data.append(imgs)
        test_labels.append(labels)
    test_data = torch.cat(test_data, 0).cpu().numpy().reshape(len(test_loader.dataset), -1)
    test_labels = torch.cat(test_labels, 0).cpu().numpy()

# --- YOUR XGBOOST CODE ---
print("\nStarting XGBoost Training...")
model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=0)
model.fit(train_data, train_labels)

# Evaluation
test_preds = model.predict(test_data)
accuracy = np.mean(test_preds == test_labels)
print(f"XGBoost Final Test Accuracy: {accuracy:.4f}")