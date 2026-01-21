import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

class ResNet18(nn.Module):
    def __init__(self, pretrained=False, probing=False):
        super(ResNet18, self).__init__()
        # Initialize the ResNet18 backbone
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
            self.resnet18 = resnet18(weights=weights)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ToTensor()
            ])            
            self.resnet18 = resnet18()
                
        in_features_dim = self.resnet18.fc.in_features        
        self.resnet18.fc = nn.Identity()
        
        # Freeze layers if linear probing is selected
        if probing:
            for name, param in self.resnet18.named_parameters():
                param.requires_grad = False
        
        # Final classification layer
        self.logistic_regression = nn.Linear(in_features_dim, 1)

    def forward(self, x):
        features = self.resnet18(x)
        x = self.logistic_regression(features)
        return x

def get_loaders(path, transform, batch_size):
    # Prepare data loaders for train, validation and test sets
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(root=os.path.join(path, 'train'), transform=transform),
        batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(root=os.path.join(path, 'val'), transform=transform),
        batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(root=os.path.join(path, 'test'), transform=transform),
        batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def compute_accuracy(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            predicted = (outputs.squeeze() > 0).long()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def visualize_discrepancies(best_model, worst_model, test_loader, device, num_images=5):
    # Find 5 samples where the best model is correct and worst model is wrong
    best_model.eval()
    worst_model.eval()
    found_images = []
    
    print("\nSearching for discrepancy samples for report visualization...")
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            best_preds = (best_model(imgs).squeeze() > 0).long()
            worst_preds = (worst_model(imgs).squeeze() > 0).long()
            
            for i in range(len(labels)):
                if best_preds[i] == labels[i] and worst_preds[i] != labels[i]:
                    img_display = imgs[i].cpu().permute(1, 2, 0).numpy()
                    # Rescale to [0,1] for display
                    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
                    found_images.append((img_display, labels[i].item()))
                
                if len(found_images) >= num_images: break
            if len(found_images) >= num_images: break

    plt.figure(figsize=(15, 5))
    for i, (img, true_label) in enumerate(found_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        label_text = "Real" if true_label == 1 else "Fake"
        plt.title(f"True: {label_text}\nFine-tuned: OK\nScratch: Fail")
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Path configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'whichfaceisreal')

    # Mode 1: pretrained=False, probing=False (Scratch)
    # Mode 2: pretrained=True, probing=True   (Probing)
    # Mode 3: pretrained=True, probing=False  (Fine-tuning)
    IS_PRETRAINED = True 
    IS_PROBING = False
    
    model = ResNet18(pretrained=IS_PRETRAINED, probing=IS_PROBING).to(device)

    try:
        train_loader, val_loader, test_loader = get_loaders(data_path, model.transform, 32)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # Training loop
        print(f"Running Experiment: Pretrained={IS_PRETRAINED}, Probing={IS_PROBING}")
        for epoch in range(3):
            model.train()
            for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs).squeeze(), labels.float())
                loss.backward()
                optimizer.step()
            
            val_acc = compute_accuracy(model, val_loader, device)
            print(f"Val Accuracy: {val_acc:.4f}")

        # Final Test Result
        test_acc = compute_accuracy(model, test_loader, device)
        print(f"\nFINAL TEST ACCURACY: {test_acc:.4f}")

        # Visualization Section
        if not IS_PRETRAINED:
            torch.save(model.state_dict(), "scratch_model.pth")
            print("Successfully saved scratch_model.pth for comparison.")
        
        if IS_PRETRAINED and not IS_PROBING:
            # If we just finished Fine-tuning, try to compare with saved Scratch model
            scratch_file = "scratch_model.pth"
            if os.path.exists(scratch_file):
                worst_model = ResNet18(pretrained=False, probing=False).to(device)
                worst_model.load_state_dict(torch.load(scratch_file))
                visualize_discrepancies(model, worst_model, test_loader, device)
            else:
                print("Note: scratch_model.pth not found. Run Scratch mode first to enable visualization.")

    except Exception as e:
        print(f"An error occurred: {e}")