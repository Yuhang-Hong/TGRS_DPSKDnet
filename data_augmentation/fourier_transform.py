import torch
import numpy as np


def FAM(X, a=0.8, b=1.1, beta_mean=0.1, beta_std=0.05):
    """
    对输入张量 X 进行傅里叶变换，在幅度和相位上添加乘性和加性噪声，并返回经过傅里叶逆变换后的张量。

    参数:
    - X: 输入张量 (形状: [batch_size, channels, height, width])
    - a, b: 用于生成乘性噪声系数 alpha 的区间 [a, b]
    - beta_mean, beta_std: 用于生成加性噪声 beta 的均值和标准差

    返回:
    - 带噪声的逆傅里叶变换结果，形状与输入 X 相同
    """
    device = X.device  # 获取输入张量的设备（CPU 或 GPU）

    # 对输入张量进行傅里叶变换
    # X_fft = torch.fft.fft2(X)
    X_fft = torch.fft.fft2(X, norm="ortho")

    # 分离出幅度和相位
    amplitude = torch.abs(X_fft)  # 幅度
    phase = torch.angle(X_fft)    # 相位

    # 在 [a, b] 区间内为每个元素采样 alpha
    alpha = (a + (b - a) * torch.rand(X.shape, device=device)).to(device)

    # 从正态分布 N(beta_mean, beta_std^2) 中为每个元素采样 beta_noise
    beta_noise = torch.randn(X.shape, device=device) * beta_std + beta_mean
    # print(beta_noise)
    # 添加乘性和加性噪声到幅度和相位
    amplitude_noise = alpha * amplitude + beta_noise
    phase_noise = alpha * phase + beta_noise

    # 根据修改后的幅度和相位重建傅里叶变换结果
    real_part = amplitude_noise * torch.cos(phase_noise)
    imag_part = amplitude_noise * torch.sin(phase_noise)
    X_fft_noisy = torch.complex(real_part, imag_part)

    # 进行傅里叶逆变换
    # X_ifft = torch.fft.ifft2(X_fft_noisy).real

    X_ifft = torch.fft.ifft2(X_fft_noisy, norm="ortho").real
    # 将结果转换为 float32 类型
    X_ifft = X_ifft.float()

    return X_ifft


if __name__ == "__main__":
    # 判断 GPU 是否可用，并将张量放在 GPU 上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建一个随机的输入张量，并放在 GPU 上
    X = torch.randn(256, 48, 13, 13).to(device)

    # 调用函数，添加噪声
    X_noisy = FAM(X)

    # 打印输出形状和数据类型
    print(X_noisy.shape)  # 输出: torch.Size([256, 48, 13, 13])
    print(X.dtype)
    print(X_noisy.dtype)
