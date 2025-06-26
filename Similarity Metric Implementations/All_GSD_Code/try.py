from torchvision import models
from timm import create_model
import loralib as lora
import torch.nn as nn

model = create_model(
            'vit_tiny_patch16_224',
            num_classes=14,
            apply_lora=True,
            lora_r=1,
            lora_alpha=1,
            pretrained=True
        )
model.head = nn.Sequential(model.head, nn.Sigmoid())
print(model)
lora.mark_only_lora_as_trainable(model)
in_features = model.head.in_features
# model.head = nn.Sequential(nn.Linear(in_features, num_classes), nn.Sigmoid())
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(params)