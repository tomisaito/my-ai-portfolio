import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

def create_model(num_classes, pretrained=True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None

    model = models.resnet18(weights=weights)
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model
