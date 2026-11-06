"""Model construction and seeding shared by training and evaluation."""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

import config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_model():
    """ImageNet-pretrained ResNet-18 with a 7-way head."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(config.CLASSES))
    return model.to(config.DEVICE)
