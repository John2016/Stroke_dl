function [ train_x, train_y, test_x, test_y ] = preprocess_4( input_data )
%PREPROCESS_4 Transfer the y to 3-bytes: positive negative or   not-sure (the inconsistent data)
%   Detailed explanation goes here
[nrow, ncol] = size(input_data);

%% 3-byte
load('Inconsistent');
index_delete = [index_delete(:,1), index_delete(:,2)];
new_y = ones([nrow, 3]);
for ii = 1:nrow
   %negative
    if input_data(ii,1) == 99
        new_y(ii,:) = [0 1 0];
    %positive
    else
        new_y(ii,:) = [1 0 0];
    end
end
%inconsistent
for jj = 1:length(index_delete)
    new_y(index_delete(jj),:) = [0 0 1];
end

%% normalization
input_data =mapminmax(input_data',0,1)';

%% random
rand_index = randperm(nrow);
input_data = input_data(rand_index,:);
new_y = new_y(rand_index,:);

%% seperate train & test
n_train = floor(nrow / 10000) * 10000;
train_x = double(input_data(1:n_train, 2:end));
train_y = double(new_y(1:n_train, :));
test_x = double(input_data((n_train + 1:end), 2:end));
test_y = double(new_y((n_train + 1:end), :));

end

