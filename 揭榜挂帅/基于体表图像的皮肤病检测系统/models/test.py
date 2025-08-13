import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from main import ImprovedSwinUNet  # 从你的训练代码里导入模型类

# ------------------------
# 1. 路径设置（两个图）
# ------------------------
visible_path = r"D:\projects\image_division\DSC00005.JPG"  # 普通光图片
uv_path = r"D:\projects\image_division\DSC00006.JPG"       # 伍德灯图片
model_path = "swinunet_lesion.pth"

# ------------------------
# 2. 图像预处理
# ------------------------
transform_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ------------------------
# 3. 加载模型
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ImprovedSwinUNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ------------------------
# 4. 预测函数
# ------------------------
def predict_mask(image_path, threshold=0.3):
    img = Image.open(image_path).convert("RGB")
    tensor_img = transform_img(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor_img, target_size=(224, 224))
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    return (pred_mask > threshold).astype(np.uint8), img  # 返回掩码 & 原图

def calculate_area(mask):
    return np.sum(mask)

# ------------------------
# 5. 预测普通光 & 伍德灯
# ------------------------
mask_vis, img_vis = predict_mask(visible_path)
mask_uv, img_uv = predict_mask(uv_path)

area_vis = calculate_area(mask_vis)
area_uv = calculate_area(mask_uv)

status = "进展期" if area_uv > area_vis else "稳定期"

print(f"普通光面积: {area_vis} | 伍德灯面积: {area_uv} | 判断: {status}")

# ------------------------
# 6. 可视化
# ------------------------
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
axs[0, 0].imshow(img_vis)
axs[0, 0].set_title("普通光原图")
axs[0, 0].axis("off")

axs[0, 1].imshow(mask_vis, cmap="gray")
axs[0, 1].set_title("普通光掩码")
axs[0, 1].axis("off")

axs[1, 0].imshow(img_uv)
axs[1, 0].set_title("伍德灯原图")
axs[1, 0].axis("off")

axs[1, 1].imshow(mask_uv, cmap="gray")
axs[1, 1].set_title("伍德灯掩码")
axs[1, 1].axis("off")

plt.tight_layout()
plt.show()
