%% train a deep NN using the SAE model
function [ nn, auc ] = train_dbn( train_x, train_y, test_x, test_y)

%%  to train a 40-70 hidden units DBN and use its weights to initialize a NN
rand('state',0)
%train dbn
dbn.sizes = [40];
opts.numepochs = 20;
opts.batchsize = 2;
opts.momentum = 0;
opts.alpha = 0.008;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

rand('state',0)
dbn.sizes = [40 70];
opts.numepochs =   25;
opts.batchsize = 2;
opts.momentum  =   0;
opts.alpha     =   0.008;
opts.plot = 1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

% unfold dbn to nn
nn = dbnunfoldtonn(dbn, 1);
nn.activation_function = 'sigm';        % ? only the output-layer's activation function
nn.learningRate = 5;
nn.scaling_learningRate = 0.99;

%figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights
opts.numepochs =  500;
opts.batchsize = 20;
opts.plot = 1;
nn = nntrain(nn, train_x, train_y, opts);
saveas(gcf,'nntrain.jpg');
[auc, predict] = nntest(nn, test_x, test_y);
%save nn

end
