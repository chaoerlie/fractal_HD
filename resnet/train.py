import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import models, transforms
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 导入 tqdm 用于显示进度条
import argparse

# 1. 自定义数据集类
class HausdorffDataset(data.Dataset):
    def __init__(self, image_paths, hd_values, transform=None):
        self.image_paths = image_paths
        self.hd_values = hd_values
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.hd_values[idx]  # Hausdorff维数标签

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


# # 2. 数据预处理和加载
# image_folder = 'dataset'  # 训练图片存放的文件夹路径
# hd_file = 'hd_values.txt'  # 存放HD标签的文件路径

# # 读取所有图片路径
# image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if
#                f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]

# # 读取Hausdorff维数标签
# hd_values = np.loadtxt(hd_file)

# # 定义数据预处理（增加数据增强）
# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),  # 随机水平翻转
#     transforms.RandomRotation(10),  # 随机旋转 10 度
#     transforms.Resize((224, 224)),  # 缩放到224x224大小
#     transforms.ToTensor(),  # 转换为 Tensor
# ])

# # 将数据集分为训练集和验证集（80%用于训练，20%用于验证）
# train_image_paths, val_image_paths, train_hd_values, val_hd_values = train_test_split(image_paths, hd_values, test_size=0.2, random_state=42)

# # 创建训练集和验证集
# train_dataset = HausdorffDataset(train_image_paths, train_hd_values, transform)
# val_dataset = HausdorffDataset(val_image_paths, val_hd_values, transform)

# train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# 3. 构建ResNet152 + 多尺度卷积模型
class MultiScaleResNet152(nn.Module):
    def __init__(self):
        super(MultiScaleResNet152, self).__init__()

        # 加载ResNet152预训练模型
        self.resnet152 = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)  # 使用 weights 替代 pretrained
        self.resnet152.fc = nn.Identity()  # 移除ResNet152的全连接层

        # 多尺度卷积头
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=3, padding=1)  # 3x3卷积
        self.conv2 = nn.Conv2d(2048, 512, kernel_size=5, padding=2)  # 5x5卷积
        self.conv3 = nn.Conv2d(2048, 512, kernel_size=7, padding=3)  # 7x7卷积
        
        # 回归头，输出一个值表示Hausdorff维数
        self.fc = nn.Linear(512 * 3, 1)  # 拼接多尺度特征后送入全连接层进行回归

    def forward(self, x):
        # ResNet152特征提取
        # 检查输入形状，以兼容图像输入和特征输入
        if x.dim() == 4: # [B, C, H, W]
            features = self.resnet152(x)  # 输出形状：[batch_size, 2048]
        else: # [B, D]
            features = x

        features = features.view(features.size(0), -1, 1, 1) # 形状变为 [batch_size, 2048, 1, 1]

        # 多尺度卷积提取不同尺度的特征
        scale1 = self.conv1(features)
        scale2 = self.conv2(features)
        scale3 = self.conv3(features)

        # 拼接多尺度特征
        combined_features = torch.cat((scale1, scale2, scale3), dim=1)

        # 使用全连接层进行回归
        output = self.fc(combined_features.view(combined_features.size(0), -1))

        return output


# # 4. 设置训练参数
# model = MultiScaleResNet152()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)  # 确保模型在正确的设备上

# # 损失函数
# criterion = nn.MSELoss()

# # 优化器
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# # 学习率调度器
# scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # 每5个epoch将学习率降低为原来的0.1

# # TensorBoard 可视化
# writer = SummaryWriter(log_dir='./logs')

# # 5. 训练模型
# epochs = 400
# save_interval = 20  # 每20个epoch保存一次模型
# for epoch in range(epochs):
#     start_epoch_time = time.time()  # 记录epoch开始时间
#     model.train()
#     running_loss = 0.0
    
#     # 使用tqdm显示进度条
#     for batch_idx, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}'):
#         images, labels = images.to(device), labels.to(device)  # 将数据移动到设备上

#         optimizer.zero_grad()
#         outputs = model(images)

#         loss = criterion(outputs.squeeze(), labels)  # 去掉输出的多余维度
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#         # 每100个batch输出一次损失
#         if (batch_idx + 1) % 100 == 0:
#             print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

#     avg_loss = running_loss / len(train_loader)
#     print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}")

#     # 记录到 TensorBoard
#     writer.add_scalar('Loss/train', avg_loss, epoch)

#     # 每n个epoch保存模型
#     if (epoch + 1) % save_interval == 0:
#         model_save_path = f"model_epoch_{epoch + 1}.pth"
#         torch.save(model.state_dict(), model_save_path)
#         print(f"Model saved to {model_save_path}")

#     # 更新学习率
#     scheduler.step()

#     epoch_time = time.time() - start_epoch_time  # 计算每个epoch的时间
#     print(f"Epoch [{epoch + 1}/{epochs}] completed in {epoch_time:.2f} seconds.")

# print("Training Finished!")

def preprocess_image_to_tensor(image_path, size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),  # 归一化到 [0,1]
    ])
    return transform(img).unsqueeze(0)  # [1, C, H, W]

def load_resnet_hd_model(model_path=None, device=None):
    device = torch.device(device) if device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = MultiScaleResNet152().to(device)
    if model_path:
        ckpt = torch.load(model_path, map_location=device)
        state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        try:
            model.load_state_dict(state)
        except RuntimeError:
            new_state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(new_state)
    model.eval()
    return model, device

def infer_image_hd(image_path, model_path=None, device=None):
    model, device = load_resnet_hd_model(model_path, device)
    img_t = preprocess_image_to_tensor(image_path).to(device)
    with torch.no_grad():
        pred = model(img_t)  # 期望形状 [1] 或 [1,1]
    return float(pred.view(-1).item())

if __name__ == '__main__':
    parser = argparse.ArgumentParser('HD predictor (simple)')
    parser.add_argument('--image_path', type=str, default="/home/ps/zhw/fractal_HD/train/output/imagenet/ariter64,16-temp1.03-linearcfg11.0-filter0.0001-image64_20250422_183252/00011.png", help='Path to input image')
    parser.add_argument('--model_path', type=str, default="/home/ps/zhw/fractal_HD/resnet/model_epoch_400.pth", help='Path to trained HD model (.pth). If omitted, model initialized with pretrained ResNet weights and random head.')
    parser.add_argument('--device', type=str, default='cuda', help='device string, e.g. "cpu" or "cuda:0"')
    args = parser.parse_args()

    hd_value = infer_image_hd(args.image_path, args.model_path, args.device)
    print(f"Predicted HD for {args.image_path}: {hd_value:.6f}")
