import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from model.unet_model import UNet  # 根据实际路径调整

# 参数设置
input_path = r"D:\裂缝数据集\CrackDataset-main\crack500\val"
output_path = r"D:\裂缝数据集\输出图像"
model_path = r"D:\裂缝识别项目\checkpoints\final_model_1.pth"
input_size = 256
THRESHOLD = 0.5  # 二值化阈值


def predict_mask(model, image_path, output_path, device="cuda", img_size=256, threshold=0.5):

    # 灰度值
    org_image = Image.open(image_path).convert('L')

    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    input_tensor = preprocess(org_image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)
        pred_mask = (probs > threshold).float()

    mask_np = pred_mask.squeeze().cpu().numpy().astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_np).resize(
        org_image.size, resample=Image.NEAREST)

    os.makedirs(output_path, exist_ok=True)
    base_name = os.path.basename(image_path).split('.')[0]
    mask_img.save(os.path.join(output_path, f"{base_name}.png"))


def main():
    device = torch.device("cuda")

    model = UNet(n_channels=1, n_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    img_files = [f for f in os.listdir(input_path)]
    for f in img_files:
        predict_mask(model, os.path.join(input_path, f),
                     output_path, device, input_size, THRESHOLD)
    print("Finished!")


if __name__ == "__main__":
    main()
