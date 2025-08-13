import argparse
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from main import ImprovedSwinUNet  # 只导入模型，不会触发训练代码

def predict_mask(model, image_path, device, threshold=0.3):
    transform_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    tensor_img = transform_img(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor_img, target_size=(224, 224))
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    return (pred_mask > threshold).astype(np.uint8)

def calculate_area(mask):
    return np.sum(mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--result_csv", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImprovedSwinUNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    results = []
    with open(args.data_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            wood_path = row["Wood_path"]
            normal_path = row["Normal_path"]

            mask_wood = predict_mask(model, wood_path, device)
            mask_normal = predict_mask(model, normal_path, device)

            area_wood = calculate_area(mask_wood)
            area_normal = calculate_area(mask_normal)

            status = "进展期" if area_wood > area_normal else "稳定期"
            results.append([idx, status])

    with open(args.result_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(results)

    print(f"✅ 推理完成，结果已保存到 {args.result_csv}")
