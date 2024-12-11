import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import label_binarize

# Directories
input_dir = r"E:\rishita\Lung\lung_image_set_LC\model"  # Update with your data directory
folders = ['lung_aca', 'lung_n', 'lung_scc']  # Normalized folders for the three classes
models_dir = r"E:\rishita\code\trainmean_std\model"  # Existing folder for saving models
results_dir = r"E:\rishita\code\trainmean_std\train"

# Hyperparameters
batch_size = 64
num_classes = len(folders)
epochs = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset
class LungDataset(Dataset):
    def __init__(self, input_dir, folders, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform

        for label, folder in enumerate(folders):
            folder_path = os.path.join(input_dir, folder)
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                self.images.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Data Transformations with Normalization
transform = transforms.Compose([
    transforms.Resize((384, 384)),  # Resize the image to 384x384
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Dataset
dataset = LungDataset(input_dir, folders, transform=transform)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 192, 192)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (64, 96, 96)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (128, 48, 48)
        )
        
        # Correct flattened size: 128 channels with 48x48 feature maps
        flattened_size = 128 * 48 * 48

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Metrics and plotting functions
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy, all_preds, all_labels

def plot_metrics(history, fold_no):
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot Training and Validation Loss
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, history['train_loss'], label="Train Loss")
    plt.plot(epochs, history['val_loss'], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve for Fold {fold_no}")
    plt.legend()
    plt.savefig(os.path.join(results_dir, f"loss_curve_fold_{fold_no}.png"))
    plt.close()

    # Plot Training and Validation Accuracy
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, history['train_acc'], label="Train Accuracy")
    plt.plot(epochs, history['val_acc'], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve for Fold {fold_no}")
    plt.legend()
    plt.savefig(os.path.join(results_dir, f"accuracy_curve_fold_{fold_no}.png"))
    plt.close()

def plot_heatmap(cm, fold_no):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=folders, yticklabels=folders)
    plt.title(f"Confusion Matrix for Fold {fold_no}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(results_dir, f"confusion_matrix_fold_{fold_no}.png"))
    plt.close()

def plot_auc_curve(fpr, tpr, fold_no):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC Curve for Fold {fold_no}")
    plt.savefig(os.path.join(results_dir, f"roc_curve_fold_{fold_no}.png"))
    plt.close()

# Initialize variables for the best model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_model = None
best_accuracy = 0
best_fold = 0

for fold_no, (train_idx, val_idx) in enumerate(kf.split(dataset), start=1):
    print(f"\n--- Fold {fold_no} ---")

    # Split dataset
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = CNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Store metrics
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    results = []

    # Training
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / len(train_loader.dataset)

        val_loss, val_acc, val_preds, val_labels = evaluate_model(model, val_loader, criterion)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save model
    model_path = os.path.join(models_dir, f"model_fold_{fold_no}.pth")
    torch.save(model.state_dict(), model_path)

    # Save results
    df_results = pd.DataFrame({'Actual': val_labels, 'Predicted': val_preds})
    df_results.to_csv(os.path.join(results_dir, f"results_fold_{fold_no}.csv"), index=False)

    # Metrics
    report = classification_report(val_labels, val_preds, target_names=folders, output_dict=True)
    cm = confusion_matrix(val_labels, val_preds)
    plot_metrics(history, fold_no)
    plot_heatmap(cm, fold_no)

    # AUC Calculation
    val_labels_bin = label_binarize(val_labels, classes=[0, 1, 2])
    val_preds_bin = label_binarize(val_preds, classes=[0, 1, 2])
    
    # Compute AUC
    fpr, tpr, _ = roc_curve(val_labels_bin.ravel(), val_preds_bin.ravel())
    plot_auc_curve(fpr, tpr, fold_no)

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        best_model = model
        best_fold = fold_no

print(f"\nBest Model: Fold {best_fold} with Accuracy: {best_accuracy:.4f}")
torch.save(best_model.state_dict(), os.path.join(models_dir, "best_model.pth"))

