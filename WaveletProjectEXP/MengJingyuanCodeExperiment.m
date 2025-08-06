clear; close all; clc;

%% 1. 读取图像并转为灰度
A_original = imread("Siakam.png");
if size(A_original, 3) == 3
    A = im2double(rgb2gray(A_original)); % RGB转灰度
else
    A = im2double(A_original); % 直接处理灰度图
end
figure("Name", "Original Image"); imshow(A); 
title('Original Image');

%% 2. 扩展图像尺寸至2的整数幂（兼容DWT）
[m, n] = size(A);
new_m = 2^ceil(log2(m)); % 扩展高度
new_n = 2^ceil(log2(n)); % 扩展宽度
A_padded = padarray(A, [new_m - m, new_n - n], 255, 'post'); % 填充白色
figure("Name", "Padded Image"); imshow(A_padded);
title('Padded Image to 2^N Size');

%% 3. 离散小波变换（DWT）与能量守恒验证
% 使用Haar小波进行单层分解
[cA, cH, cV, cD] = dwt2(A_padded, 'haar');
W = [cA, cH; cV, cD]; % 小波系数矩阵

% 验证能量守恒
E_original = sum(A_padded(:).^2);
E_transformed = sum(cA(:).^2) + sum(cH(:).^2) + sum(cV(:).^2) + sum(cD(:).^2);
energy_error = abs(E_original - E_transformed);
fprintf('Energy Conservation Validation:\n');
fprintf('  Original Energy: %.4f\n', E_original);
fprintf('  Transformed Energy: %.4f\n', E_transformed);
fprintf('  Energy Error: %.4e (Should be close to 0)\n', energy_error);

%% 4. 基于阈值的压缩
lambda = 0.5; % 压缩阈值
W_compressed = W .* (abs(W) > lambda); % 硬阈值压缩

% 计算压缩率
compression_ratio = nnzabs(W_compressed) / numel(W_compressed);
fprintf('Compression Ratio: %.2f%%\n', (1 - compression_ratio) * 100);

%% 5. 逆DWT重构图像
A_reconstructed = idwt2(...
    W_compressed(1:size(cA,1), 1:size(cA,2)), ... % cA
    W_compressed(1:size(cH,1), size(cA,2)+1:end), ... % cH
    W_compressed(size(cA,1)+1:end, 1:size(cV,2)), ... % cV
    W_compressed(size(cA,1)+1:end, size(cV,2)+1:end), ... % cD
    'haar');

% 裁剪回原始尺寸
A_reconstructed = A_reconstructed(1:m, 1:n);

%% 6. 结果显示
figure("Name", "Compressed Image");
imshow(A_reconstructed);
title(sprintf('Compressed Image (\\lambda=%.1f)', lambda));

% 计算重构误差
reconstruction_error = sum((A(:) - A_reconstructed(:)).^2);
fprintf('Reconstruction MSE: %.4e\n', reconstruction_error);

%% 7. 能量保留分析
E_retained = sum(A_reconstructed(:).^2) / E_original * 100;
fprintf('Energy Retained: %.2f%%\n', E_retained);