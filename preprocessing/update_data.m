%% 加大脑卒中病例在训练数据集中的比例
function [ data_updated ] = update_data(data, times)
%UPDATE_DATA Summary of this function goes here
%   Detailed explanation goes here
data_cuzhong = data(data(:,1)==1,:);
data_updated = data;
for i = 2:times
    data_updated = [data_updated; data_cuzhong];
end
[nrow, ncol] = size(data_updated);
data_updated = data_updated(randperm(nrow), : );
end

