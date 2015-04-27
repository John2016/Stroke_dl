function [ auc,nn ] = logistic_regression( input_data )
%LOGISTIC_REGRESSION As the baseline
%   Detailed explanation goes here
[train_x, train_y, test_x, test_y] = preprocess_3(update_data(input_data, 3));
rand('state',0)
nn = nnsetup([86 1]);
nn.learningRate = 1;   
nn.activation_function = 'sigm';    %  Sigmoid activation function
nn.scaling_learningRate = 0.99;

opts.numepochs =  50;   %  Number of full sweeps through data
opts.batchsize = 20;  %  Take a mean gradient step over this many samples
opts.plot = 1;
[nn, L] = nntrain(nn, train_x, train_y, opts);
figure; bar(nn.W{1}(2:end));
[auc, pre] = nntest(nn, test_x, test_y);
[auc_all,pre] = nntest(nn,[train_x;test_x],[train_y;test_y]);
%[auc_left,pre] = nntest(cut_nn(nn,1:54),[train_x;test_x],[train_y;test_y]);
auc = [auc,auc_all,auc_left];
%er
%assert(er < 0.08, 'Too big error');
end

