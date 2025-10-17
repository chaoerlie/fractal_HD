import torch
import torch.utils.data as data
from torchvision import models, transforms
from PIL import Image
import numpy as np

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


# 2. 数据预处理和加载
image_folder = 'dataset'  # 训练图片存放的文件夹路径
hd_file = 'hd_values.txt'  # 存放HD标签的文件路径

# 读取所有图片路径
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if
               f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]

# 读取Hausdorff维数标签
hd_values = np.loadtxt(hd_file)

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 创建数据集
dataset = HausdorffDataset(image_paths, hd_values, transform)
test_loader = data.DataLoader(dataset, batch_size=32, shuffle=False)

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
        features = self.resnet152(x)  # 输出形状：[batch_size, 2048]

        # 将特征形状调整为 [batch_size, 2048, 1, 1]，使其适应卷积层
        features = features.unsqueeze(2).unsqueeze(3)  # 形状变为 [batch_size, 2048, 1, 1]

        # 多尺度卷积提取不同尺度的特征
        scale1 = self.conv1(features)
        scale2 = self.conv2(features)
        scale3 = self.conv3(features)

        # 拼接多尺度特征
        combined_features = torch.cat((scale1, scale2, scale3), dim=1)

        # 使用全连接层进行回归
        output = self.fc(combined_features.view(combined_features.size(0), -1))

        return output


# 4. 加载模型检查点
model = MultiScaleResNet152()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

checkpoint_path = 'model_epoch_50.pth'  # 这里是训练过程中保存的模型文件路径
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)  # 加载模型权重

# 5. 测试模型
model.eval()  # 切换到评估模式
with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        predictions = model(images).squeeze()
        for i in range(len(predictions)):
            print(f"True HD: {targets[i].item():.3f}, Predicted HD: {predictions[i].item():.3f}")

print("Testing Finished!")
