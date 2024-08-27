import torch
import torch.nn as nn
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

# 加载模型的函数
def load_model(model_path, device):
    model = CNNModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 评估模型的函数
def evaluate_model(model, data_loader, device):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():  # 禁用梯度计算
        for inputs, labels in data_loader:  # 遍历测试集
            inputs = inputs.to(device)  # 将输入数据移动到设备
            label_list = [list(label) for label in labels]  # 真实标签列表
            labels = torch.stack([torch.tensor([ord(c) - ord('0') if c.isdigit() else ord(c) - ord('a') + 10 for c in label]) for label in labels])
            labels = labels.to(device)  # 将标签移动到设备

            outputs = model(inputs)  # 前向传播
            _, preds = torch.max(outputs, 2)  # 获取预测结果
            preds = preds.cpu().numpy()  # 将预测结果移动到CPU并转换为numpy数组

            for i in range(len(labels)):
                true_label = ''.join([chr(c + ord('0')) if c < 10 else chr(c + ord('a') - 10) for c in labels.cpu().numpy()[i]])
                pred_label = ''.join([chr(c + ord('0')) if c < 10 else chr(c + ord('a') - 10) for c in preds[i]])
                all_labels.append(true_label)
                all_preds.append(pred_label)
                print(f'真实值: {true_label}, 预测值: {pred_label}')
                if true_label == pred_label:
                    correct += 1
                total += 1

    accuracy = correct / total  # 计算准确率
    print(f'准确率: {accuracy:.4f}')  # 打印准确率
    return accuracy, all_labels, all_preds

def main():
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((100, 300)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
    ])

    batch_size = 32  # 设置批处理大小

    # 创建预测数据集和数据加载器
    predict_dataset = CaptchaDataset(img_dir="D:\\pycharm file\\Leave CAPTCHA Recognition\\datasets\\predict", transform=transform)
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备（CPU或GPU）

    # 加载模型
    model_path = 'captcha_model.pth'
    model = load_model(model_path, device)

    # 评估模型
    evaluate_model(model, predict_loader, device)

if __name__ == "__main__":
    main()
