import io
from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import CNNModel

app = Flask(__name__)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = CNNModel()
model.load_state_dict(torch.load('captcha_model.pth', map_location=device))
model.to(device)
model.eval()

# 图像变换
transform = transforms.Compose([
    transforms.Resize((100, 300)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, preds = torch.max(output, 2)
        preds = preds.squeeze().cpu().numpy()

    # 将预测结果转换为验证码字符串
    captcha = ''.join([chr(p + ord('0')) if p < 10 else chr(p - 10 + ord('a')) for p in preds])

    return jsonify({'captcha': captcha})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=19999)
