import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm  # 引入tqdm进度条库

# 自定义数据集类，用于加载验证码图片及其标签
class CaptchaDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir  # 图像目录路径
        self.transform = transform  # 图像变换
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]  # 获取目录中所有PNG文件

    def __len__(self):
        return len(self.img_files)  # 返回数据集大小

    def __getitem__(self, idx):
        img_name = self.img_files[idx]  # 获取图像文件名
        label = img_name[:4]  # 文件名前四个字符作为标签
        img_path = os.path.join(self.img_dir, img_name)  # 图像文件路径
        image = Image.open(img_path)  # 打开图像

        if self.transform:
            image = self.transform(image)  # 应用图像变换

        return image, label  # 返回图像及其标签


# 定义一个位置编码器，用于Transformer模型
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


# 定义Transformer模型
class TransformerCaptchaModel(nn.Module):
    def __init__(self, img_size=(100, 300), patch_size=10, num_classes=36, d_model=512, nhead=8, num_layers=6):
        super(TransformerCaptchaModel, self).__init__()
        self.patch_size = patch_size  # 图像被分割成的patch大小
        self.d_model = d_model

        # 图像转换为 patch 的 token
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)  # 计算总的patch数量
        self.patch_embedding = nn.Linear(patch_size * patch_size * 3, d_model)  # 每个patch展平后映射到 d_model 维度

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer 编码器
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 输出层
        self.fc = nn.Linear(d_model, 4 * num_classes)  # 输出每个字符的分类，共4个字符，每个字符有num_classes个类别

    def forward(self, x):
        batch_size = x.size(0)

        # 1. 将输入图像拆分为 patch
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, -1, self.patch_size * self.patch_size * 3)  # 将图像展平为序列

        # 2. 对 patch 进行线性映射并加入位置编码
        x = self.patch_embedding(x)  # 将每个patch映射到 d_model 维度
        x = x.transpose(0, 1)  # Transformer expects input with shape (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)  # 添加位置编码

        # 3. 使用 Transformer 编码器进行处理
        x = self.transformer_encoder(x)

        # 4. 提取分类结果
        x = x.mean(dim=0)  # 对所有patch进行平均池化，得到一个全局表示
        x = self.fc(x)  # 通过全连接层，输出4个字符，每个字符有num_classes个类别
        x = x.view(batch_size, 4, -1)  # 输出形状为 (batch_size, 4, num_classes)

        return x


# 训练模型的函数，使用 tqdm 来显示进度条
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0

        # tqdm 用来显示进度条
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch_idx, (inputs, labels) in progress_bar:
            inputs = inputs.to(device)
            # 标签转换为数字
            labels = torch.stack(
                [torch.tensor([ord(c) - ord('0') if c.isdigit() else ord(c) - ord('a') + 10 for c in label]) for label
                 in labels])
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs.view(-1, 36), labels.view(-1))  # 计算损失
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # 更新 tqdm 进度条上的损失信息
            progress_bar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'第 {epoch}/{num_epochs - 1} 轮, 平均损失: {epoch_loss:.4f}')
    return model


# 评估模型的函数
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = torch.stack(
                [torch.tensor([ord(c) - ord('0') if c.isdigit() else ord(c) - ord('a') + 10 for c in label]) for label
                 in labels])
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 2)
            preds = preds.cpu().numpy()

            for i in range(len(labels)):
                if all(preds[i] == labels.cpu().numpy()[i]):
                    correct += 1
                total += 1

    accuracy = correct / total
    print(f'准确率: {accuracy:.4f}')


def main():
    transform = transforms.Compose([
        transforms.Resize((100, 300)),
        transforms.ToTensor(),
    ])

    batch_size = 128
    train_dataset = CaptchaDataset(img_dir='D:\pycharm file\Leave CAPTCHA Recognition\datasets\\train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = CaptchaDataset(img_dir='D:\pycharm file\Leave CAPTCHA Recognition\datasets\\test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerCaptchaModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train_model(model, train_loader, criterion, optimizer, device, num_epochs=25)
    evaluate_model(model, test_loader, device)

    torch.save(model.state_dict(), 'captcha_transformer_model.pth')
    print('模型已保存到 captcha_transformer_model.pth')


if __name__ == "__main__":
    main()
