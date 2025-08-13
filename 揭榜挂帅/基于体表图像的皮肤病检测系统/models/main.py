import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

# ------------------------ #
# 数据集类
# ------------------------ #
class LesionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_img=None, transform_mask=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transform_img = transform_img
        self.transform_mask = transform_mask

        # 检查缺失的 mask
        missing = []
        for img_name in self.image_files:
            mask_name = Path(img_name).stem + ".png"
            if not os.path.exists(os.path.join(mask_dir, mask_name)):
                missing.append(img_name)
        if missing:
            print(f"⚠ 缺少掩码的图片：{missing}")
            self.image_files = [f for f in self.image_files if f not in missing]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = Path(img_name).stem + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        mask = (mask > 0.5).float()
        return image, mask

# ------------------------ #
# Swin-UNet 模型
# ------------------------ #
class ImprovedSwinUNet(nn.Module):
    def __init__(self):
        super(ImprovedSwinUNet, self).__init__()
        self.encoder = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            features_only=True
        )

        # decoder
        self.up4 = nn.ConvTranspose2d(768, 384, 2, stride=2)
        self.conv4 = nn.Conv2d(384 + 384, 384, 3, padding=1)

        self.up3 = nn.ConvTranspose2d(384, 192, 2, stride=2)
        self.conv3 = nn.Conv2d(192 + 192, 192, 3, padding=1)

        self.up2 = nn.ConvTranspose2d(192, 96, 2, stride=2)
        self.conv2 = nn.Conv2d(96 + 96, 96, 3, padding=1)

        self.up1 = nn.ConvTranspose2d(96, 32, 2, stride=2)
        self.conv1 = nn.Conv2d(32, 32, 3, padding=1)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, target_size=None):
        feats = self.encoder(x)
        feats = [f.permute(0, 3, 1, 2) for f in feats]  # NHWC → NCHW

        f1, f2, f3, f4 = feats[0], feats[1], feats[2], feats[3]

        x = self.up4(f4)
        x = torch.cat([x, f3], dim=1)
        x = F.relu(self.conv4(x))

        x = self.up3(x)
        x = torch.cat([x, f2], dim=1)
        x = F.relu(self.conv3(x))

        x = self.up2(x)
        x = torch.cat([x, f1], dim=1)
        x = F.relu(self.conv2(x))

        x = self.up1(x)
        x = F.relu(self.conv1(x))

        x = self.final_conv(x)

        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x

# ------------------------ #
# 训练流程
# ------------------------ #
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    transform_mask = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = LesionDataset(
        image_dir="test_image",  # 训练集路径
        mask_dir="json_png",     # 训练集mask路径
        transform_img=transform_img,
        transform_mask=transform_mask
    )
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = ImprovedSwinUNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 30
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs, target_size=masks.shape[2:])
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "swinunet_lesion.pth")
    print("✅ 训练完成，模型已保存！")
