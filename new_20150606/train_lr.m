function [ nn,auc ] = train_lr( train_x,train_y,test_x,test_y )
%LOGISTIC_REGRESSION As the baseline
%   To deal with it as a NN without hidden layer
rand('state',0)
nn = nnsetup([70 1]);
nn.learningRate = 0.005;   
nn.activation_function = 'sigm';    %  Sigmoid activation function
nn.scaling_learningRate = 0.997;

opts.numepochs =  50;   %  Number of full sweeps through data

opts.batchsize = 1;  %  Take a mean gradient step over this many samples
opts.plot = 1;
[nn, L] = nntrain(nn, train_x, train_y, opts);
%figure; bar(nn.W{1}(2:end));
[auc, pre] = nntest(nn, test_x, test_y);
end
