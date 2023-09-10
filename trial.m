clc;
clear all;
close all;

load('train_labels.mat');
load('train_features.mat');
load('test_labels.mat');
load('test_features.mat');

n1=sum(train_labels==0);
n5=sum(train_labels==1);
data=train_features;

normal = 'D:\Project Pattern\archive\chest_xray\trainresize\NORMAL\';
pneumo= 'D:\Project Pattern\archive\chest_xray\trainresize\PNEUMONIA\';






data=data';




data1=data(:,1:n1);
data2=data(:,n1+1:n1+n5);


labels=zeros(1,n1+n5);
labels(1:n1)=0;
labels(n1+1:n1+n5)=1;


mu_1 = mean(data1,2);
mu_5 = mean(data2,2);
d = size(data,1);
[W,Z,lamda] = pca(data,length(data(:,1))); % Priniciple Basis Vectors from both digit
disp(size(W))


%%%%%%%%% part 7 and 8%%%%%%%%

for i = 1 : n1
x1 = data1(:,i);
z1(i) = W(:,end-2)'*(x1 - mu_1); % First Priniciple Component
z1_2(i) = W(:,end-1)'*(x1 - mu_1);
z1_3(i) = W(:,end)'*(x1 - mu_1);


end

for i=1:n5
x5 = data2(:,i);
z5(i) = W(:,end-2)'*(x5 - mu_5); % First Priniciple Component
z5_2(i) = W(:,end-1)'*(x5 - mu_5);
z5_3(i) = W(:,end)'*(x5 - mu_5);

end

figure()
scatter(z1,zeros(size(z1)),'r');
hold on

scatter(z5,zeros(size(z5)),'b');
title('First principle component for Digit 1 and 5')
legend('Digit 1 component', 'Digit 5 component');

figure()
scatter(z1,z1_2,'r')
hold on
scatter(z5,z5_2,'b')

figure()
scatter3(z1,z1_2,z1_3,'b')
hold on
scatter3(z5,z5_2,z5_3,'r')




%%%%%%%%%%%% Part 9 and 10 %%%%%%%%%%%


dim1 = size(data1,1);
dim5 = size(data2,1);
Sigma1 = zeros(dim1,dim1);
Sigma5 = zeros(dim5,dim5);

%sigma of 1
for i = 1 : n1
x1 = data1(:,i);
Sigma1 = Sigma1 + (x1 - mu_1)*(x1 - mu_1)';
end

%sigma of 5
for i=1:n5
x5 = data2(:,i);
Sigma5 = Sigma5 + (x5 - mu_5)*(x5 - mu_5)';
end

Sigma_Within = Sigma1 + Sigma5; % within Class

Sigma_Between = (mu_1 - mu_5)*(mu_1 - mu_5)';% Between Class


w = pinv(Sigma_Within)*(mu_1 - mu_5);  %Fisher discriminant vector


for i = 1 : n1
x1 = data1(:,i);
yk1(i) = w'*x1; % Digit 1 FDA Component
end

for i=1:n5
x5 = data2(:,i);
yk5(i) = w'*x5; % Digit 5 FDA Component

end



figure()
scatter(yk1,zeros(size(yk1)),'r');
hold on
scatter(yk5,zeros(size(yk5)),'b');
title('Fisher Discriminant Components of Digit 1 and Digit 5')
legend('Digit 1 component', 'Digit 5 component');





function data=dataload(image_folder)
% Read image files
%image_folder = 'C:\Users\nahia\Google Drive (nahian.buet11@gmail.com)\Fall-21 Semester Drive Folder\ECSE 6610 Pattern\HW\HW 7 2021\hw7data\train_data\'; % Change
%this to your directory location
file_pattern = fullfile(image_folder, '*.jpeg');
image_files = dir(file_pattern);
nfiles = length(image_files);
data = zeros(nfiles, 64*64); % Matrix with vectorized images along rows
filename_num = zeros(nfiles, 1); % Vector of filenames
for i = 1:nfiles
filename = image_files(i).name;
filename_num(i) = str2double(filename(1:(end - 5)));
im = imread([image_folder, filename]); % Read i th image
im=imresize(im, [64 64]);

data(i, :) = reshape(im, [1,64*64]); % Save image along i th row
end
[filename_num, order] = sort(filename_num); % Sort filenames
data = data(order, :); % Rearrange data matrix to correct order
data = data / 255;% Divide each pixel by 255



end




