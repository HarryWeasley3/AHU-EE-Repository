clc
clear
close all

%% 设置参数
img = imread('image.jpg');
img = rescale(img);
theta = -30;

%% 最近邻插值
near = imrotate(img,theta,"nearest","crop");
figure
subplot(1,2,1)
imshow(img)
title("原始图像")
subplot(1,2,2)
imshow(near)
title("最近邻插值")
% saveas(gcf, '最近邻插值.jpg')
%% 双线性插值
bil = imrotate(img,theta,"bilinear","crop");
figure
subplot(1,2,1)
imshow(img)
title("原始图像")
subplot(1,2,2)
imshow(bil)
title("双线性插值")
% saveas(gcf, '双线性插值.jpg')
%% 双三次插值
cub = imrotate(img,theta,"bicubic","crop");
figure
subplot(1,2,1)
imshow(img)
title("原始图像")
subplot(1,2,2)
imshow(cub)
title("双三次插值")
% saveas(gcf, '双三次插值.jpg')

%% 对比
figure
subplot(2,2,1)
imshow(img)
title("原始图像")
subplot(2,2,2)
imshow(near)
title("最近邻插值")
subplot(2,2,3)
imshow(bil)
title("双线性插值")
subplot(2,2,4)
imshow(cub)
title("双三次插值")