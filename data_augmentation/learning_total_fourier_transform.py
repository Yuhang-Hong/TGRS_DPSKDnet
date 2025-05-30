import torch
import torch.nn as nn
import numpy as np


class FourierTransformModule(nn.Module):
    def __init__(self, a=0.5, b=0.8, beta_mean=0.1, beta_std=0.05):
        super(FourierTransformModule, self).__init__()

        # 初始化可学习参数
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        self.beta_mean = nn.Parameter(
            torch.tensor(beta_mean, dtype=torch.float32))
        self.beta_std = nn.Parameter(
            torch.tensor(beta_std, dtype=torch.float32))

    def forward(self, X):
        """
        对输入张量 X 的所有通道应用傅里叶变换，并添加噪声。
        """
        device = X.device
        batch_size, channels, height, width = X.shape

        # 使用 clamp 限制 a 和 b 的范围
        a = torch.clamp(self.a, min=0.0, max=1.0)
        b = torch.clamp(self.b, min=a.item(), max=1.0)

        # beta_mean 和 beta_std 使用 clamp 限制范围
        beta_mean = torch.clamp(self.beta_mean, min=0.0, max=0.5)
        beta_std = torch.clamp(self.beta_std, min=0.0, max=0.2)

        # 对所有通道应用傅里叶变换
        X_fft = torch.fft.fft2(X, norm='ortho')

        # 分离出幅度和相位
        amplitude = torch.abs(X_fft)
        phase = torch.angle(X_fft)

        # 在 [a, b] 区间内为每个通道采样 alpha
        alpha = (a + (b - a) * torch.rand(batch_size,
                 channels, 1, 1, device=device))

        # 从正态分布 N(beta_mean, beta_std^2) 中为每个通道采样 beta_noise
        beta_noise = torch.randn(batch_size, channels,
                                 1, 1, device=device) * beta_std + beta_mean

        # 添加乘性和加性噪声到幅度和相位
        amplitude_noise = alpha * amplitude + beta_noise
        phase_noise = alpha * phase + beta_noise

        # 根据修改后的幅度和相位重建傅里叶变换结果
        real_part = amplitude_noise * torch.cos(phase_noise)
        imag_part = amplitude_noise * torch.sin(phase_noise)
        X_fft_noisy = torch.complex(real_part, imag_part)

        # 进行傅里叶逆变换并取实部
        X_ifft = torch.fft.ifft2(X_fft_noisy, norm='ortho').real.float()

        return X_ifft


if __name__ == "__main__":
    # 判断 GPU 是否可用，并将张量放在 GPU 上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建一个随机的输入张量，并放在 GPU 上
    X = torch.randn(256, 48, 13, 13).to(device)

    # 创建模型实例
    model = FourierTransformModule().to(device)

    # 打印可学习参数
    for name, param in model.named_parameters():
        print(f"{name}: {param.item()}")

    # 调用模型，对所有通道执行傅里叶变换
    X_transformed = model(X)

    # 打印输出形状和数据类型
    print(X_transformed.shape)  # 输出: torch.Size([256, 48, 13, 13])
    print(X.dtype)
    print(X_transformed.dtype)

