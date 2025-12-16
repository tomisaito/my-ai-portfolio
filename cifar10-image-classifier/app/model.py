import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path

# CIFAR-10 class labels
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# SimpleCNN model definition (must match training-time architecture)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Preprocessing pipeline
def get_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Model loader
def load_model(model_files=None):
    if model_files is None:
        model_files = ['cifar10_model_fixed.pth', 'cifar10_model.pth']

    BASE_DIR = Path(__file__).resolve().parent 
    
    model = SimpleCNN()
    model_loaded = False

    print("📦 Loading model...")

    for model_file in model_files:
        try:
            file_path = BASE_DIR / model_file
            
            print(f"  Trying: {file_path}")
            
            checkpoint = torch.load(file_path, map_location=torch.device('cpu')) 

            # Handle both types of checkpoints
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            model_loaded = True
            print(f"✅ Loaded: {model_file}")
            break
        except FileNotFoundError:
            print(f"  ⚠️ File not found: {file_path}")
        except Exception as e:
            print(f"  ⚠️ Load failure: {e}")

    if not model_loaded:
        print("❌ Model load failed. Prediction will not work.")

    return model, model_loaded