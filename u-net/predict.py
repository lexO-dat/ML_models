import torch
from torchvision import transforms
from train_test import device, decode_segmap
import UNet
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def preprocess_image(image_path):
    
    transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor()
    ])

    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor, original_size, image

def predict_custom_image(model, image_path):
    # preprocess the image
    input_tensor, original_size, original_image = preprocess_image(image_path)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # (256, 256)

    # Resize prediction to original image dimensions
    pred_original_res = np.array(Image.fromarray(pred.astype(np.uint8)).resize(
        original_size,
        Image.NEAREST  # Preserve label IDs
    ))

    return pred_original_res, original_image

def visualize_prediction(original_image, prediction):
    # Convert prediction to color mask
    color_mask = decode_segmap(prediction)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(color_mask)
    ax[1].set_title("Predicted Mask")
    ax[1].axis('off')

    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(n_classes=21).to(device)
model.load_state_dict(torch.load("./models/best_unet.pth", map_location=device))
model.eval()

pred_mask, orig_img = predict_custom_image(model, "./images/enriqueta.jpeg")
visualize_prediction(orig_img, pred_mask)