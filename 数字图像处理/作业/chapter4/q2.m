clc
clear
close all

img = double(imread("img/image.png"));
% img = rgb2gray(img);
img = img./255;
img = rgb2gray(img);
% img = img/max(img(:))* 255;
% img = log(double(img) + 1);
figure
imshow(img)
title("Original Image")
size(img)

% FFT
lnimg = log(img+eps);
Freq = fftshift(fft2(lnimg));

% 同态滤波参数设置
d = 20;     % 截止频率

alphaL = 0.5; % 调整低频增益
alphaH = 1.2; % 调整高频增益

[r,c] = size(Freq);
center = [r/2,c/2];
LPF = zeros(r,c);
for i = 1:r
    for j = 1:c
        dist = sqrt((i-center(1))^2 + (j - center(2))^2);
        LPF(i,j) = exp((-dist^2)/d^2);
    end
end
HPF = (alphaH-alphaL)*(1-LPF)+alphaL;

figure
mesh(HPF)

% 同态滤波;
filtered_F = HPF.*Freq;
% filtered_F = imgaussfilt(Freq);
filtered_img = abs(ifft2(ifftshift(filtered_F)));
filtered_img = exp(filtered_img);
filtered_img = mat2gray(filtered_img);
filtered_img = 1-filtered_img;
figure
subplot(1,2,1)
imshow(img)
title("Original Image")
subplot(1,2,2)
imshow(filtered_img)
title("Filtered Image") 