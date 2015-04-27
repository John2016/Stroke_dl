%% keep the target to be 1-byte
function [ train_x, train_y, test_x, test_y ] = preprocess_2( input_data )
%PREPROCESS_2 Summary of this function goes here
%   Detailed explanation goes here
% perform random
[nrow, ncol] = size(input_data);
input_data = input_data(randperm(nrow),:);

% 1-0 target
input_data(input_data(:,1) == 99,1) = 0;

% seperate train & test
n_train = floor(nrow / 10000) * 10000;
train_x = double(input_data(1:n_train, 2:end));
train_y = double(input_data(1:n_train, 1));
test_x = double(input_data((n_train + 1:end), 2:end));
test_y = double(input_data((n_train + 1:end), 1));

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);
end

