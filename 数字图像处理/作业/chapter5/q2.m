clc
clear
close all

% img = imread ("img/blurred.jpg");
% img = rgb2gray(img);
img = imread("img/cameraman.tif");

% figure
% imshow(img)

%% 逆滤波器
T = 1;
a = 0.1;
b = 0.1;

freq = fft2(img);
freq = fftshift(freq);
amp = abs(freq);
phase = angle(freq);

[r,s] = size(freq);
u0 = floor(s/2+1);
v0 = floor(r/2+1);
[jj,ii] = meshgrid(1:s,1:r);
u = ii - u0;
v = jj - v0;

% 构造H(u,v)
den = (u*a+v*b);
H = (T./(pi.*den)).*sin(pi.*den).*exp(-1j.*pi.*den);
H(den==0) = T;
figure
imshow(abs(H))
title("运动模糊的传递函数")
% 构造被运动模糊的图像
blurred_freq = H.*freq;
blurred_img = ifft2(ifftshift(blurred_freq));
blurred_img = mat2gray(abs(blurred_img));
% figure
% imshow(blurred_img)

% 无噪声时
recover_f = blurred_freq./H;
recover_f = ifftshift(recover_f);
recover = ifft2(recover_f);
recover = mat2gray(abs(recover));
% figure
% imshow(recover); 

% 空域添加高斯噪声
n = sqrt(1); % 标准差
noise = n.*randn(size(blurred_freq));
noised_freq = blurred_freq + noise; % 添加高斯噪声
noised_img = mat2gray(abs(ifft2(ifftshift(noised_freq)))); %加噪后的运动模糊图像
noised_recover_freq = noised_freq./H;
noised_recover_img = ifft2(ifftshift(noised_recover_freq));
noised_recover_img = mat2gray(abs(noised_recover_img));
% figure
% imshow(noised_recover_img) % 发现噪声被放大
% 限制H不会过小
threshold = 0.01;
H(abs(H)<threshold) = threshold;

noised_recover_freq = noised_freq./H;
noised_recover_img2 = ifft2(ifftshift(noised_recover_freq));
noised_recover_img2 = mat2gray(abs(noised_recover_img2));
% figure
% imshow(noised_recover_img2)
figure
subplot(1,2,1)
imshow(img)
title("原始图像")
subplot(1,2,2)
imshow(blurred_img)
title("运动模糊后的图像")

figure
subplot(2,2,1)
imshow(recover)
title("无噪声时恢复图像")
subplot(2,2,2)
imshow(noised_img)
title("加噪后的运动模糊图像")
subplot(2,2,3)
imshow(noised_recover_img)
title("有噪声时恢复图像")
subplot(2,2,4)
imshow(noised_recover_img2)
title("限制H后的恢复图像")

%% 维纳滤波
F_hat = (1./H).*((abs(H)^2)./(abs(H)^2 + n^2)).*noised_freq;
winner_img = ifft2(ifftshift(F_hat));
winner_img = mat2gray(abs(winner_img));
figure
subplot(1,2,1)
imshow(noised_img)
title("加噪后的运动模糊图像")
subplot(1,2,2)
imshow(winner_img)
title("维纳滤波后的图像")

