import torch as torch
import torch.nn as nn
import torch.nn.functional as F

class yolo(nn.Module):
    def __init__(self, num_classes=21, anchors=None, num_anchors=3):
        super(yolo, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors if anchors is not None else [(10,13), (16,30), (33,23)]
        self.num_anchors = num_anchors

        # Note: Original YOLO uses 20 classes for PASCAL VOC, but keeping your parameter flexible
        # For PASCAL VOC: S=7, B=2, C=20, output = 7x7x30
        
        self.convs = nn.Sequential(
            # Layer 1: Conv 7x7x64-s-2
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2: Conv 3x3x192
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3-5: Conv layers with 1x1 reduction
            nn.Conv2d(192, 128, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 6-13: Conv layers (repeated 1x1 and 3x3 pattern x4)
            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 14-20: Conv layers (repeated 1x1 and 3x3 pattern x2)
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Layer 21-22: Conv layers
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Layer 23-24: Final conv layers
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Fully connected layers
        # Output size for PASCAL VOC: 7*7*30 = 1470
        # General formula: S*S*(B*5 + C) where S=7, B=2, C=num_classes
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 7 * 7 * (2 * 5 + self.num_classes)),  # S*S*(B*5+C)
        )
    
    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x)
        # Reshape to (batch_size, S, S, B*5+C)
        x = x.view(-1, 7, 7, 2 * 5 + self.num_classes)
        return x