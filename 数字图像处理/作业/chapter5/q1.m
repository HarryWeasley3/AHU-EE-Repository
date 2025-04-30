clc
clear 
close all

img = imread('fMOST/fMOST鼠脑图1.png');
img = rgb2gray(img);

freq = fft2(img);
freq = fftshift(freq); % 将频谱中心化
A = abs(freq); % 取幅值
phase = angle(freq); % 取相位
logA = log(eps + A); % 取对数幅值
logA = mat2gray(logA); % 归一化幅值
phase = mat2gray(phase); % 归一化相位
figure
subplot(1,2,1)
imshow(logA)
title('幅值谱')
subplot(1,2,2)
imshow(phase)
title('相位谱')
 

% 陷波滤波器
[r, c] = size(A);
D0 = 6;
D1 = 3;
filter1 = ones(r, c);
for d = 1:r

    if d > r / 2 - D0 + 1 && d < r / 2 + D0
        filter1(d,:) = 0;
    end

end

for i = 1:c

    if i > floor(c / 2 - D1 +1 ) && i < floor(c / 2 + D1 +1)
        filter1(:, i) = 1;
    end

end
filter2 = ~filter1;
% A = A.*filter1; % 频谱乘以滤波器
freq = freq.*filter1; % 频谱乘以滤波器

figure
subplot(1,2,1)
imshow(log(abs(freq)) + eps)
img_filtered = ifft2(ifftshift(freq));
img_filtered = real(img_filtered); % 取实部 
img_filtered = mat2gray(img_filtered); % 归一化

freq2 = filter2.*A;
subplot(1,2,2)
imshow(log(abs(freq2')) + eps)
noise = ifft2(ifftshift(freq2));
noise = real(noise); % 取实部 
noise = mat2gray(noise); % 归一化


figure
subplot(1,2,1)
imshow(img)
title('原图像')
subplot(1,2,2)
imshow(img_filtered)
% title('滤波后图像')

figure
imshow(noise)
% title("噪声模式")