%% separate the test set from the universal set thoroughly 
[nrow, ncol] = size(input_data);
input_data = input_data(randperm(nrow),:);

% 1-0 target
input_data(input_data(:,1) == 99,1) = 0;

% 0-1 normalize
input_data(:,2:end) = mapminmax(input_data(:,2:end)',0,1)';
%train_x = mapminmax(train_x',0,1)';
%test_x = mapminmax(test_x',0,1)';

% seperate train & test
n_train = floor(nrow / 10000) * 10000;
train_x = double(input_data(1:n_train, 2:end));
train_y = double(input_data(1:n_train, 1));
test_x = double(input_data((n_train + 1:end), 2:end));
test_y = double(input_data((n_train + 1:end), 1));
