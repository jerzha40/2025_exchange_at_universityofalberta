clear; close all; clc;

%% 1. 读取图像并转为灰度
A_original = imread("lena.g"); % 替换为你的图片路径
if size(A_original, 3) == 3
    A = im2double(rgb2gray(A_original)); % RGB转灰度
else
    A = im2double(A_original); % 直接处理灰度图
end
figure("Name", "Original Image"); imshow(A); 
title('Original Image');

%% 2. 计算原始图像的能量（平方和）
energy_original = sum(A(:).^2);
fprintf('原始图像能量 (E_original): %.6f\n', energy_original);

%% 3. 扩展图像尺寸至2的整数幂（兼容DWT）
[m, n] = size(A);
new_m = 2^ceil(log2(m)); % 扩展高度
new_n = 2^ceil(log2(n)); % 扩展宽度
A_padded = padarray(A, [new_m - m, new_n - n], 0, 'post'); % 填充黑色（避免引入额外能量）
figure("Name", "Padded Image"); imshow(A_padded);
title('Padded Image to 2^N Size');

%% 4. 离散小波变换（DWT）分解
% 使用Haar小波进行单层分解
[cA, cH, cV, cD] = dwt2(A_padded, 'haar');

%% 5. 计算分解后各子带的能量
energy_cA = sum(cA(:).^2); % 近似子带（低频）
energy_cH = sum(cH(:).^2); % 水平细节子带
energy_cV = sum(cV(:).^2); % 垂直细节子带
energy_cD = sum(cD(:).^2); % 对角细节子带
energy_transformed = energy_cA + energy_cH + energy_cV + energy_cD;

fprintf('分解后各子带能量:\n');
fprintf('  cA (低频): %.6f\n', energy_cA);
fprintf('  cH (水平细节): %.6f\n', energy_cH);
fprintf('  cV (垂直细节): %.6f\n', energy_cV);
fprintf('  cD (对角细节): %.6f\n', energy_cD);
fprintf('分解后总能量 (E_transformed): %.6f\n', energy_transformed);

%% 6. 验证能量守恒
energy_error = abs(energy_original - energy_transformed);
fprintf('能量守恒验证:\n');
fprintf('  |E_original - E_transformed| = %.6e\n', energy_error);
if energy_error < 1e-10
    fprintf('✅ 能量守恒成立 (误差可忽略)\n');
else
    fprintf('❌ 能量守恒不成立！需检查填充或变换过程\n');
end

%% 7. 可视化子带系数（可选）
figure;
subplot(2,2,1); imshow(cA, []); title('Approximation (cA)');
subplot(2,2,2); imshow(cH, []); title('Horizontal Detail (cH)');
subplot(2,2,3); imshow(cV, []); title('Vertical Detail (cV)');
subplot(2,2,4); imshow(cD, []); title('Diagonal Detail (cD)');

%% 8. 重构图像（可选）
A_reconstructed = idwt2(cA, cH, cV, cD, 'haar');
A_reconstructed = A_reconstructed(1:m, 1:n); % 裁剪回原始尺寸
figure; imshow(A_reconstructed); title('Reconstructed Image');
reconstruction_error = sum((A(:) - A_reconstructed(:)).^2);
fprintf('重构误差 (MSE): %.6e\n', reconstruction_error);