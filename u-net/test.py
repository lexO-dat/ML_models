import torch
from dataset import train_loader, val_loader
import matplotlib.pyplot as plt
from train_test import device, model
import numpy as np

def decode_segmap(mask):
    """Convert mask IDs to color image."""
    # VOC color map
    label_colors = np.array([
        (0, 0, 0),       # 0=background
        (128, 0, 0),     # 1=aeroplane
        (0, 128, 0),     # 2=bicycle
        (128, 128, 0),   # 3=bird
        (0, 0, 128),     # 4=boat
        (128, 0, 128),   # 5=bottle
        (0, 128, 128),   # 6=bus
        (128, 128, 128), # 7=car
        (64, 0, 0),      # 8=cat
        (192, 0, 0),     # 9=chair
        (64, 128, 0),    # 10=cow
        (192, 128, 0),   # 11=dining table
        (64, 0, 128),    # 12=dog
        (192, 0, 128),   # 13=horse
        (64, 128, 128),  # 14=motorbike
        (192, 128, 128), # 15=person
        (0, 64, 0),      # 16=potted plant
        (128, 64, 0),    # 17=sheep
        (0, 192, 0),     # 18=sofa
        (128, 192, 0),   # 19=train
        (0, 64, 128)     # 20=tv/monitor
    ])
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)

    for l in range(0, 21):
        idx = mask == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

# Take a batch and visualize
model.eval()
with torch.no_grad():
    images, masks = next(iter(train_loader))
    images, masks = images.to(device), masks.to(device)
    outputs = model(images)
    preds = torch.argmax(outputs, dim=1)

# Convert to numpy for plotting
images = images.cpu().permute(0, 2, 3, 1).numpy()
masks = masks.cpu().numpy()
preds = preds.cpu().numpy()

for i in range(len(images)):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(images[i])
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(decode_segmap(masks[i]))
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(decode_segmap(preds[i]))
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    plt.show()