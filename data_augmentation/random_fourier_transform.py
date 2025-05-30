import torch
import numpy as np


def split_and_apply_fourier(X, split_ratio=0.5, a=0.7, b=1.0, beta_mean=0.1, beta_std=0.05):
    """
    对输入张量 X 在通道维度上进行随机拆分，一部分通道应用傅里叶变换并添加噪声，另一部分通道保持不变，最后组合。

    参数:
    - X: 输入张量 (形状: [batch_size, channels, height, width])
    - split_ratio: 应用傅里叶变换的通道占比，例如 0.5 表示一半的通道进行傅里叶变换
    - a, b: 用于生成乘性噪声系数 alpha 的区间 [a, b]
    - beta_mean, beta_std: 用于生成加性噪声 beta 的均值和标准差

    返回:
    - 经过随机拆分和傅里叶变换组合后的张量，形状与输入 X 相同
    """
    device = X.device  # 获取输入张量的设备（CPU 或 GPU）
    batch_size, channels, height, width = X.shape

    # 生成随机掩码，仅在通道维度进行拆分
    # mask 为 shape: [batch_size, channels, 1, 1]，扩展到所有空间维度
    mask = (torch.rand(batch_size, channels, 1, 1,
            device=device) < split_ratio).float()

    # 对掩码为1的通道应用傅里叶变换
    X_fft = torch.fft.fft2(X, norm='ortho')

    # 分离出幅度和相位
    amplitude = torch.abs(X_fft)  # 幅度
    phase = torch.angle(X_fft)    # 相位

    # 在 [a, b] 区间内为每个通道采样 alpha
    alpha = (a + (b - a) * torch.rand(batch_size,
             channels, 1, 1, device=device)).to(device)

    # 从正态分布 N(beta_mean, beta_std^2) 中为每个通道采样 beta_noise
    beta_noise = torch.randn(batch_size, channels, 1,
                             1, device=device) * beta_std + beta_mean

    # 添加乘性和加性噪声到幅度和相位
    amplitude_noise = alpha * amplitude + beta_noise
    phase_noise = alpha * phase + beta_noise

    # 根据修改后的幅度和相位重建傅里叶变换结果
    real_part = amplitude_noise * torch.cos(phase_noise)
    imag_part = amplitude_noise * torch.sin(phase_noise)
    X_fft_noisy = torch.complex(real_part, imag_part)

    # 进行傅里叶逆变换并取实部
    X_ifft = torch.fft.ifft2(X_fft_noisy, norm='ortho').real.float()

    # 根据掩码组合傅里叶变换部分和未变换部分
    # mask 为1时选择 X_ifft，否则选择 X
    X_combined = X_ifft * mask + X * (1 - mask)

    return X_combined


if __name__ == "__main__":
    # 判断 GPU 是否可用，并将张量放在 GPU 上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建一个随机的输入张量，并放在 GPU 上
    X = torch.randn(256, 48, 13, 13).to(device)

    # 调用函数，进行通道维度的随机拆分和傅里叶变换
    X_transformed = split_and_apply_fourier(X)

    # 打印输出形状和数据类型
    print(X_transformed.shape)  # 输出: torch.Size([256, 48, 13, 13])
    print(X.dtype)
    print(X_transformed.dtype)
