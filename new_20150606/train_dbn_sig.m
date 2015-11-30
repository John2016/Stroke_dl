%% train a deep NN using the SAE model
function [ nn, auc ] = train_dbn_sig( train_x, train_y, test_x, test_y)
train_y = train_y(:,2);
test_y = test_y(:,2);
%%  to train a 40-70 hidden units DBN and use its weights to initialize a NN
rand('state',0)
%train dbn
dbn.sizes = [150];
opts.numepochs = 30;
opts.batchsize = 20;
opts.momentum = 0.8;
opts.alpha = 0.008;
opts.plot = 1; 
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

rand('state',0)
dbn.sizes = [150 70];
opts.numepochs =   30;
opts.batchsize = 20;
opts.momentum  =   0.8;
opts.alpha     =   0.008;
opts.plot = 1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

rand('state',0)
%dbn.sizes = [150 70 35];
dbn.sizes = [150 70 70];
opts.numepochs =   30;
opts.batchsize = 20;
opts.momentum  =   0.8;
opts.alpha     =   0.008;
opts.plot = 1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

% unfold dbn to nn
nn = dbnunfoldtonn(dbn, 1);
nn.activation_function = 'tanh_opt';        % ? only the output-layer's activation function
nn.learningRate = 6;
%nn.scaling_learningRate = 0.99;
nn.scaling_learningRate = 0.9993;
nn.output = 'sigm';
nn.dropoutFraction = 0.1;
nn.inputZeroMaskedFraction = 0.05;
%figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights
%opts.numepochs =  500;
opts.numepochs =  500;
opts.batchsize = 100;
opts.plot = 1;
nn = nntrain(nn, train_x, train_y, opts);
%saveas(gcf,'nntrain.jpg');
[auc, predict] = nntest(nn, test_x, test_y);
%save nn

end
