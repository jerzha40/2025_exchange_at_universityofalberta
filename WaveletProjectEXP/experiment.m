clear;close all; clc;
A = imread("pic_1.jpg");
A = im2double(rgb2gray(A));
figure("Name","原图"); imshow(A); % 查看灰度图像创建情况

judge = 1; % 压缩力度

[m, n] = size(A);
% 先扩展为2的整次幂大小
tempm = ceil2(m); tempn = ceil2(n);
A = [A, 255*ones(m, tempn-n); 255*ones(tempm - m, n), 255*ones(tempm - m, tempn-n)];
figure("Name","拓展图"); imshow(A); 
M1 = basevecM_D(tempm); % 建立单位正交基
M2 = basevecM_D(tempn);
coffi = M1*A*(M2'); % 系数矩阵
coffi = coffi.*(abs(coffi) > judge); % 压缩
B = (M1')*coffi*M2;
figure("Name","压缩拓展图"); imshow(B);

B = B(1:m, 1:n);
figure("Name","压缩图"); imshow(B);

function AnsM = basevecM_D(num) % 单位化

    AnsM = orthvecM_D(num);
    len = size(AnsM, 1);
    for i = 1:len
        AnsM(i,:) = AnsM(i,:)./sqrt(AnsM(i,:)*(AnsM(i,:)'));
    end
    
end


function AnsM = orthvecM_D(num) % 建立正交基
    AnsM = orthvecM_D1(num);
    AnsM = [ones(1, num); AnsM];
end

function AnsM = orthvecM_D1(num) 

    if num == 2
        AnsM = [1, -1];
    else 
        if mod(num, 2) == 0
            tempM1 = orthvecM_D1(num/2);
            tempM2 = zeros(size(tempM1, 1), num/2);
            tempM3 = ones(1, num/2);
            AnsM = [tempM3, -tempM3; tempM1, tempM2; tempM2, tempM1];
        else
            tempM1 = orthvecM_D1(num-1);
            tempM2 = zeros(size(tempM1, 1), 1);
            AnsM = [tempM1, tempM2];
        end % 图片大小不一定为2的整数次幂啊
    
    end
end

function ans1 = ceil2(x)
    ans1 = log2(x);
    ans1 = ceil(ans1);
    ans1 = pow2(ans1);
end