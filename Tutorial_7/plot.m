clear all; close all; clc

% 載入結果資料
data = load('results.txt');  % 格式: x y h_Body T

% 取出欄位
x = data(:,1);
y = data(:,2);
body = data(:,3);

% 定義網格大小
NX = 100;
NY = 100;

% 轉成 2D 矩陣
X = reshape(x, [NX, NY])';
Y = reshape(y, [NX, NY])';
Body = reshape(body, [NX, NY])';

% 畫圖
figure;
contour(X, Y, Body, [0.5 0.5], 'LineWidth', 1.5);  % 只畫出 Body=0.5 的邊界（空洞邊緣）
xlabel('CX');
ylabel('CY');
axis equal tight;
title('Body Layout (1 = solid, 0 = hole)');
