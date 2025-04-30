clc; clear; close all;

% 读取图像
img = im2double(imread('img/cameraman.tif'));
figure;
imshow(img);
title('原始图像');

% 傅里叶变换
img_fft = fftshift(fft2(img));

% 参数：噪声方差
sigma = 0.01;

%% 方法一：空域加噪声
noise_spatial = sqrt(sigma) * randn(size(img));
img_noisy_spatial = img + noise_spatial;
img_noisy_spatial_fft = fftshift(fft2(img_noisy_spatial));

%% 方法二：频域加噪声
noise_real = sqrt(sigma/2) * randn(size(img_fft));
noise_imag = sqrt(sigma/2) * randn(size(img_fft));
noise_freq = noise_real + 1i * noise_imag;
img_noisy_freq_fft = img_fft + noise_freq;
img_noisy_freq = ifft2(ifftshift(img_noisy_freq_fft));
img_noisy_freq = real(img_noisy_freq);   % 保证实数图像

%% 可视化比较

% 空域加噪后的图像
figure;
subplot(2,2,1);
imshow(img_noisy_spatial, []);
title('空域加噪后图像');

subplot(2,2,2);
imshow(log(1 + abs(img_noisy_spatial_fft)), []);
title('空域加噪后的频谱');

% 频域加噪后的图像
subplot(2,2,3);
imshow(img_noisy_freq, []);
title('频域加噪后图像');

subplot(2,2,4);
imshow(log(1 + abs(img_noisy_freq_fft)), []);
title('频域加噪后的频谱');
