import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import label_binarize
from PIL import Image

# Directories
test_dir = r"E:\rishita\Lung\lung_image_set_LC\model\test"  # Path to test folder
results_dir=r"E:\rishita\code\cnnvall\test"
models_dir = r"E:\rishita\code\cnnvall\trainmodel"  # Folder where trained models are stored

# Hyperparameters
batch_size = 64
num_classes = 3  # Lung classes: lungaca, lungn, lungscc

# Device setup for CPU-only
device = torch.device("cpu")

# Transform for test data
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define custom dataset for test data
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        self.class_map = {"lungaca": 0, "lungn": 1, "lungscc": 2}

        for img_name in os.listdir(test_dir):
            img_path = os.path.join(test_dir, img_name)
            class_name = [key for key in self.class_map if key in img_name]
            if class_name:
                self.images.append(img_path)
                self.labels.append(self.class_map[class_name[0]])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path  # Include img_path in the return


# Load test dataset
test_dataset = TestDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 48 * 48, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load the best model (CPU-compatible)
model_path = os.path.join(models_dir, "best_model_fold_2.pth")
model = CNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Evaluate on test data
# Evaluate on test data and log predictions
all_preds = []
all_labels = []
all_probs = []
all_image_paths = []  # To store image paths

with torch.no_grad():
    for inputs, labels, paths in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_image_paths.extend(paths)  # Collect corresponding image paths

# Create a DataFrame with predictions, actual labels, and image paths
results_df = pd.DataFrame({
    "Image Path": all_image_paths,
    "Actual Label": all_labels,
    "Predicted Label": all_preds
})

# Optionally, add probabilities for each class
class_names = ["lungaca", "lungn", "lungscc"]
probs_df = pd.DataFrame(all_probs, columns=[f"Prob_{cls}" for cls in class_names])
results_df = pd.concat([results_df, probs_df], axis=1)

# Save the results to a CSV file
results_csv_path = os.path.join(results_dir, "test_predictions.csv")
results_df.to_csv(results_csv_path, index=False)

print(f"Predictions saved to {results_csv_path}.")


# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["lungaca", "lungn", "lungscc"], yticklabels=["lungaca", "lungn", "lungscc"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.close()


print("Classification report")
print(classification_report(all_labels,all_preds,target_names=["lungaca","lungn","lungscc"]))

# ROC Curve and AUC
all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2])
fpr, tpr, roc_auc = {}, {}, {}
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], np.array(all_probs)[:, i])
    roc_auc[i] = roc_auc_score(all_labels_bin[:, i], np.array(all_probs)[:, i])

# Plot ROC Curves
plt.figure(figsize=(8, 6))
for i, class_name in enumerate(["lungaca", "lungn", "lungscc"]):
    plt.plot(fpr[i], tpr[i], label=f"{class_name} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(results_dir, "roc_curve.png"))
plt.close()

print("Test evaluation complete. Results saved.")
