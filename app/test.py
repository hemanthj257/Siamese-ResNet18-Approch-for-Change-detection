import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class SiameseChangeDetectionModel(nn.Module):
    def __init__(self):
        super(SiameseChangeDetectionModel, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x1, x2):
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        diff = torch.abs(f1 - f2)
        out = self.decoder(diff)
        return out

def quick_detect(image1_path, image2_path, model_path='siamese_change_detection_model_50.pth', show_plot=False):
    """
    Quick function to detect changes between two images
    Args:
        image1_path: path to first image
        image2_path: path to second image
        model_path: path to model weights
        show_plot: whether to show visualization plot
    Returns:
        numpy array of change mask
    """
    # Set up device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseChangeDetectionModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Transform for preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Load and process images
    img1 = Image.open(image1_path).convert('RGB')
    img2 = Image.open(image2_path).convert('RGB')
    img1_tensor = transform(img1).unsqueeze(0).to(device)
    img2_tensor = transform(img2).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        output = model(img1_tensor, img2_tensor)
    
    # Convert to binary mask
    change_mask = output.cpu().squeeze().numpy()
    binary_mask = (change_mask > 0.5).astype(np.float32)

    # Visualization if requested
    if show_plot:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img1)
        plt.title('Current Image')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(img2)
        plt.title('Past Image')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(binary_mask, cmap='gray')
        plt.title('Change Mask')
        plt.axis('off')
        plt.show()

    return binary_mask

# if __name__ == "__main__":
#     # Example usage
#     img1_path = "/home/hehe/final/data/validate/set 1/current.png"
#     img2_path = "/home/hehe/final/data/validate/set 1/past.png"
    
#     result = quick_detect(img1_path, img2_path, show_plot=True)
#     print(f"Change detection completed. Mask shape: {result.shape}")
