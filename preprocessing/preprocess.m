%%数据预处理，double，zscore，目标二维化，train & test，
function [ train_x, train_yy, test_x, test_yy ] = preprocess( input_data)
%PREPROCESS Summary of this function goes here
%   Detailed explanation goes here
[nrow, ncol] = size(input_data);
input_data = input_data(randperm(nrow),:);
n_train = floor(nrow / 10000) * 10000;
train_x = double(input_data(1:n_train, 2:end));
train_y = double(input_data(1:n_train, 1));
train_yy = ones(n_train,2);
for ii = 1:n_train
    if train_y(ii) == 1
        train_yy(ii,:) = [1 0];
    else
        train_yy(ii,:) = [0 1];
    end
end
test_x = double(input_data((n_train + 1:end), 2:end));
test_y = double(input_data((n_train + 1:end), 1));
test_yy = ones(length(test_y),2);
for ii = 1:length(test_y)
    if test_y(ii) == 1
        test_yy(ii,:) = [1 0];
    else
        test_yy(ii,:) = [0 1];
    end
end

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

end

