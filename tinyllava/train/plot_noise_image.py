from PIL import Image
import torchvision.transforms as transforms
from vcd_add_noise import add_diffusion_noise
# 加载图像并将其转换为Tensor
image_path = r'data/coco/train2017/000000153321.jpg'  # 替换为您的图像路径
image = Image.open(image_path).convert('RGB')
to_tensor = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像转换为Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1, 1]区间
])
image_tensor = to_tensor(image)

import torch

# 假设您的函数已经定义在当前环境中
noise = [500,600,700,750,800,850,900,950,999]
for i in noise:
    noisy_image_tensor = add_diffusion_noise(image_tensor.unsqueeze(0), i).squeeze(0)
    from torchvision.transforms import ToPILImage
    to_pil_image = ToPILImage()
    noisy_image = to_pil_image(noisy_image_tensor.clamp(-1, 1))  # 确保值在[-1, 1]区间内
    noisy_image.save(f"./noisy_image_{i}.jpg")  # 保存图像


