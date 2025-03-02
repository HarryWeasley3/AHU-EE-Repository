clc;
clear;
close all;

[x1,n1] = rec_seq(0,100,0,99);
x2 = [1, 2, 3, 4, 5];
n2 = [0,1,2,3,4];
y = OverlapAdd(x1,x2,5); % 重叠相加法计算卷积

function y = OverlapAdd(x, h, L)
M = length(h);
N = M+L-1;
% 补零
h2 = [h,zeros(1,N-M)];
% 将x分段
xi = [];
for i = 1:length(x)
    if mod(i,L) == 0
        xi = [xi;x((i/L-1)*L+1:i)];
    end
end
xi = [xi;x(floor(i/L)*L+1:end),zeros(1,L-length(x(floor(i/L)*L+1:end)))];
% 补零
xi = horzcat(xi,zeros(size(xi,1),N-L));

Xi = [];
for i = 1:size(xi,1)
    k = fft(xi(i,:));
    Xi = [Xi;k];
end

H = fft(h2);

Yi = [];
for i = 1:size(Xi,1)
    k = Xi(i,:).*H;
    Yi = [Yi;k];
end

yi = [];
for i = 1:size(Yi,1)
    k = ifft(Yi(i,:));
    yi = [yi;k];
end

y = [];
for i = 1:size(yi,1)-1
    overlap = yi(i,end-(M-1)+1:end) + yi(i+1,1:M-1);
    if i == 1
        y = [y,yi(i,1:end-(M-1)),overlap];
    else
        y = [y,yi(i,M:end-(M-1)),overlap];
    end
end
y = [y,yi(i+1,M:end)];
end

function [x, n] = rec_seq(n0, n3, n1, n2)
    % 生成 n1~n2 的矩形序列，n0, n3-1 处跳变
    n = n1:n2;
    x = ((n - n0) >= 0) - ((n - n3) >= 0);
end