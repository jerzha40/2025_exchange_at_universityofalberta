clear; close all; clc;

%% 1. 读取图像并转为灰度（确保尺寸为2的整数幂）
A_original = imread('lena.jpg'); % 替换为你的lena.jpg路径
if size(A_original, 3) == 3
    A = im2double(rgb2gray(A_original)); % RGB转灰度
else
    A = im2double(A_original);
end

% 裁剪或填充至最近的2的整数幂尺寸（避免填充引入误差）
[m, n] = size(A);
new_size = 2^floor(log2(min(m, n))); % 选择较小的维度保证正方形
A = imresize(A, [new_size, new_size]); % 强制调整为2的幂尺寸
figure; imshow(A); title('Original Image (Resized to 2^N)');

%% 2. 计算原始图像的能量
energy_original = sum(A(:).^2);
fprintf('原始图像能量 (E_original): %.8f\n', energy_original);

%% 3. 离散小波变换（DWT）分解
[cA, cH, cV, cD] = dwt2(A, 'haar');

%% 4. 计算分解后各子带的能量
energy_cA = sum(cA(:).^2);
energy_cH = sum(cH(:).^2);
energy_cV = sum(cV(:).^2);
energy_cD = sum(cD(:).^2);
energy_transformed = energy_cA + energy_cH + energy_cV + energy_cD;

fprintf('分解后各子带能量:\n');
fprintf('  cA (低频): %.8f\n', energy_cA);
fprintf('  cH (水平细节): %.8f\n', energy_cH);
fprintf('  cV (垂直细节): %.8f\n', energy_cV);
fprintf('  cD (对角细节): %.8f\n', energy_cD);
fprintf('分解后总能量 (E_transformed): %.8f\n', energy_transformed);

%% 5. 验证能量守恒
energy_error = abs(energy_original - energy_transformed);
fprintf('能量守恒验证:\n');
fprintf('  |E_original - E_transformed| = %.8e\n', energy_error);
if energy_error < 1e-8
    fprintf('✅ 能量守恒成立 (误差可忽略)\n');
else
    fprintf('❌ 能量守恒不成立！需检查图像尺寸或变换过程\n');
end

%% 6. 重构图像并计算误差
A_reconstructed = idwt2(cA, cH, cV, cD, 'haar');
reconstruction_error = sum((A(:) - A_reconstructed(:)).^2);
fprintf('重构误差 (MSE): %.8e\n', reconstruction_error);

% 显示重构图像
figure; imshow(A_reconstructed); title('Reconstructed Image');