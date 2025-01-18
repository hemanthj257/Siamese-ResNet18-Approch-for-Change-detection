import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import wandb
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Define the Siamese Network architecture
class SiameseChangeDetectionModel(nn.Module):
    def __init__(self):
        super(SiameseChangeDetectionModel, self).__init__()
        # Shared Encoder (e.g., ResNet18 without the last layers)
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last two layers
        
        # Decoder to generate the binary mask
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # 8x8 -> 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 16x16 -> 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 32x32 -> 64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # 64x64 -> 128x128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),    # 128x128 -> 256x256
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=1),               # Final layer
            nn.Sigmoid()  # For binary mask output
        )
        
    def forward(self, x1, x2):
        # Encode both images
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        # Compute feature difference
        diff = torch.abs(f1 - f2)
        # Decode to get the change mask
        out = self.decoder(diff)
        return out
    
# Prepare the dataset
class ChangeDetectionDataset(Dataset):
    def __init__(self, image_dir1, image_dir2, mask_dir, transform=None):
        self.image_dir1 = image_dir1
        self.image_dir2 = image_dir2
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Get list of files that exist in all directories
        files1 = set(os.listdir(self.image_dir1))
        files2 = set(os.listdir(self.image_dir2))
        files_mask = set(os.listdir(self.mask_dir))
        
        # Only keep files that exist in all directories
        valid_files = list(files1.intersection(files2).intersection(files_mask))
        if len(valid_files) == 0:
            raise RuntimeError(f"No matching files found in directories:\n{image_dir1}\n{image_dir2}\n{mask_dir}")
        
        print(f"Found {len(valid_files)} valid image pairs")
        self.filenames = sorted(valid_files)  # Sort to ensure consistent ordering
    
    def __len__(self):
        return len(self.filenames)  # Return the number of valid files
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # Add error handling for file loading
        try:
            img1_path = os.path.join(self.image_dir1, filename)
            img2_path = os.path.join(self.image_dir2, filename)
            mask_path = os.path.join(self.mask_dir, filename)
            
            if not (os.path.exists(img1_path) and os.path.exists(img2_path) and os.path.exists(mask_path)):
                raise FileNotFoundError(f"Missing files for {filename}")
            
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            raise
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            mask = self.transform(mask)
            mask = (mask > 0).float()
            
        return img1, img2, mask

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create dataset and dataloader
image_dir1 = '/home/hehe/final/data/Train/current'
image_dir2 = '/home/hehe/final/data/Train/past'
mask_dir = '/home/hehe/final/data/Train/masks'

dataset = ChangeDetectionDataset(image_dir1, image_dir2, mask_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

# Set up the training loop with CUDA support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseChangeDetectionModel().to(device)

# For multiple GPUs
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 50

# Initialize wandb before training
wandb.init(
    project="siamese-change-detection",
    config={
        "architecture": "Siamese-ResNet18",
        "learning_rate": 1e-4,
        "epochs": num_epochs,
        "batch_size": 8,
        "optimizer": "Adam",
        "loss_function": "BCELoss",
    }
)

# Watch the model
wandb.watch(model, criterion, log="all", log_freq=10)

# Add these metric calculation functions
def calculate_iou(pred, target):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def calculate_dice(pred, target):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)

def calculate_pixel_metrics(pred, target):
    pred = (pred > 0.5).float()
    pred_flat = pred.cpu().numpy().flatten()
    target_flat = target.cpu().numpy().flatten()
    
    precision = precision_score(target_flat, pred_flat, zero_division=0)
    recall = recall_score(target_flat, pred_flat, zero_division=0)
    f1 = f1_score(target_flat, pred_flat, zero_division=0)
    
    return precision, recall, f1

for epoch in range(num_epochs):
    model.train()
    epoch_stats = {
        'loss': 0,
        'iou': 0,
        'dice': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0
    }
    
    for batch_idx, (img1, img2, mask) in enumerate(dataloader):
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        output = model(img1, img2)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        batch_iou = calculate_iou(output, mask)
        batch_dice = calculate_dice(output, mask)
        batch_precision, batch_recall, batch_f1 = calculate_pixel_metrics(output, mask)
        
        # Update running stats
        epoch_stats['loss'] += loss.item()
        epoch_stats['iou'] += batch_iou.item()
        epoch_stats['dice'] += batch_dice.item()
        epoch_stats['precision'] += batch_precision
        epoch_stats['recall'] += batch_recall
        epoch_stats['f1'] += batch_f1
        
        # Log batch-level metrics
        if batch_idx % 10 == 0:  # Log every 10 batches
            wandb.log({
                "training/batch": batch_idx + epoch * len(dataloader),
                "training/batch_loss": loss.item(),
                "metrics/batch_iou": batch_iou.item(),
                "metrics/batch_dice": batch_dice.item(),
                "metrics/batch_precision": batch_precision,
                "metrics/batch_recall": batch_recall,
                "metrics/batch_f1": batch_f1
            })
    
    # Calculate epoch averages
    num_batches = len(dataloader)
    for key in epoch_stats:
        epoch_stats[key] /= num_batches
    
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Loss: {epoch_stats["loss"]:.4f}')
    print(f'IoU: {epoch_stats["iou"]:.4f}')
    print(f'Dice: {epoch_stats["dice"]:.4f}')
    print(f'Precision: {epoch_stats["precision"]:.4f}')
    print(f'Recall: {epoch_stats["recall"]:.4f}')
    print(f'F1: {epoch_stats["f1"]:.4f}\n')
    
    # Log epoch-level metrics
    wandb.log({
        "training/epoch": epoch,
        "training/loss": epoch_stats["loss"],
        "metrics/iou": epoch_stats["iou"],
        "metrics/dice": epoch_stats["dice"],
        "metrics/precision": epoch_stats["precision"],
        "metrics/recall": epoch_stats["recall"],
        "metrics/f1": epoch_stats["f1"]
    })

# Save model with wandb
model_path = 'siamese_change_detection_model_50_stats.pth'
torch.save(model.state_dict(), model_path)
wandb.save(model_path)

# Finish wandb run
wandb.finish()


