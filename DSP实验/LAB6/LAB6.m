clc;
clear;
close all;

[x1,n1] = rec_seq(0,100,0,99);
x2 = [1, 2, 3, 4, 5];
n2 = [0,1,2,3,4];
y = OverlapAdd(x1,x2,5); % 重叠相加法计算卷积

function [x, n] = rec_seq(n0, n3, n1, n2)
    % 生成 n1~n2 的矩形序列，n0, n3-1 处跳变
    n = n1:n2;
    x = ((n - n0) >= 0) - ((n - n3) >= 0);
end

function y = OverlapAdd(x, h, L)
    M = length(h);
    N = M + L - 1; 

    % 补零到N点
    h2 = [h, zeros(1, N - M)];

    % 将x分段
    SegNum = ceil(length(x) / L);
    xi = zeros(SegNum, N); 
    for i = 1:SegNum
        StartIdx = (i - 1) * L + 1;
        EndIdx = min(i * L, length(x));
        segment = x(StartIdx:EndIdx);
        xi(i, 1:length(segment)) = segment; % 补零
    end

    H = fft(h2);
    Xi = fft(xi, N, 2); 
    Yi = Xi .* H;
    yi = ifft(Yi, [], 2);

    % 重叠相加
    y = zeros(1, (SegNum - 1) * L + N); 
    for i = 1:SegNum
        StartIdx = (i - 1) * L + 1;
        y(StartIdx:StartIdx + N - 1) = y(StartIdx:StartIdx + N - 1) + yi(i, :);
    end
end