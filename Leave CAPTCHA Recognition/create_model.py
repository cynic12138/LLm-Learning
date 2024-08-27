import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os


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


# 定义CNN模型类
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.cnn = models.resnet18(pretrained=True)  # 使用预训练的ResNet-18模型
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 4 * 36)  # 修改最后一层，全连接层输出4*36

    def forward(self, x):
        x = self.cnn(x)  # 前向传播
        x = x.view(-1, 4, 36)  # 调整输出形状为(batch_size, 4, 36)
        return x


# 训练模型的函数
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=25):
    for epoch in range(num_epochs):  # 训练若干轮
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        for inputs, labels in train_loader:  # 遍历训练集
            inputs = inputs.to(device)  # 将输入数据移动到设备（CPU或GPU）
            # 将标签转换为数字
            labels = torch.stack(
                [torch.tensor([ord(c) - ord('0') if c.isdigit() else ord(c) - ord('a') + 10 for c in label]) for label
                 in labels])
            labels = labels.to(device)  # 将标签移动到设备

            optimizer.zero_grad()  # 清除梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs.view(-1, 36), labels.view(-1))  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item() * inputs.size(0)  # 累积损失

        epoch_loss = running_loss / len(train_loader.dataset)  # 计算平均损失
        print(f'第 {epoch}/{num_epochs - 1} 轮, 损失: {epoch_loss:.4f}')  # 打印损失
    return model


# 评估模型的函数
def evaluate_model(model, test_loader, device):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算
        for inputs, labels in test_loader:  # 遍历测试集
            inputs = inputs.to(device)  # 将输入数据移动到设备
            # 将标签转换为数字
            labels = torch.stack(
                [torch.tensor([ord(c) - ord('0') if c.isdigit() else ord(c) - ord('a') + 10 for c in label]) for label
                 in labels])
            labels = labels.to(device)  # 将标签移动到设备

            outputs = model(inputs)  # 前向传播
            _, preds = torch.max(outputs, 2)  # 获取预测结果
            preds = preds.cpu().numpy()  # 将预测结果移动到CPU并转换为numpy数组

            for i in range(len(labels)):
                if all(preds[i] == labels.cpu().numpy()[i]):  # 比较预测结果和标签
                    correct += 1
                total += 1

    accuracy = correct / total  # 计算准确率
    print(f'准确率: {accuracy:.4f}')  # 打印准确率


def main():
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((100, 300)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
    ])

    batch_size = 128  # 设置批处理大小

    # 创建训练数据集和数据加载器
    train_dataset = CaptchaDataset(img_dir='D:\pycharm file\Leave CAPTCHA Recognition\datasets\\train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 创建测试数据集和数据加载器
    test_dataset = CaptchaDataset(img_dir='D:\pycharm file\Leave CAPTCHA Recognition\datasets\\test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备（CPU或GPU）
    model = CNNModel().to(device)  # 实例化模型并移动到设备
    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 定义优化器

    # 训练模型
    model = train_model(model, train_loader, criterion, optimizer, device, num_epochs=25)
    # 评估模型
    evaluate_model(model, test_loader, device)

    # 保存模型
    torch.save(model.state_dict(), 'captcha_model.pth')
    print('模型已保存到 captcha_model.pth')


if __name__ == "__main__":
    main()
