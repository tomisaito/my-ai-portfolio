
import torch.nn as nn
from torchvision import models

def create_model(num_classes, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model
