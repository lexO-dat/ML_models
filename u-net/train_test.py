import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from PIL import Image
from UNet import UNet
from dataset import train_loader, val_loader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_classes=21).to(device)  # VOC has 21 classes (20 + background)

criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) 

# This is a common loss function for segmentation tasks (i take it from a book), it measures overlap between prediction & ground truth masks
def dice_loss(pred, target, smooth=1.):
    pred = torch.softmax(pred, dim=1)
    target_one_hot = torch.nn.functional.one_hot(target, pred.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

ce_loss = nn.CrossEntropyLoss()

def combined_loss(pred, target):
    return ce_loss(pred, target) # + dice_loss(pred, target)


best_val_loss = float("inf")
patience = 5
wait = 0

scaler = torch.cuda.amp.GradScaler()

for epoch in range(50):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    accum_steps = 1

    # Batch iteration
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = combined_loss(outputs, masks) / accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accum_steps

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] "
                  f"Batch Loss: {(loss.item()*accum_steps):.4f} | "
                  f"Avg Loss: {running_loss/(batch_idx+1):.4f}")

    train_loss = running_loss / len(train_loader)

    # Model evaluation (to see the loss and if its necessary to early stop)
    model.eval()
    val_loss = 0.0
    with torch.no_grad(), torch.cuda.amp.autocast():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)
            outputs = model(images)
            val_loss += combined_loss(outputs, masks).item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_unet.pth")
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping")
            break