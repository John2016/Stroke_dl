function [ nn, auc ] = cal_dbn( input_data)
%  calculate the auc using the DBN
%  to compare with the result made by SAE 
[train_x, train_y, test_x, test_y] = preprocess_3(update_data(input_data, 1));

%%  ex1 train a 100 hidden unit RBM and visualize its weights
rand('state',0)
dbn.sizes = [40];
opts.numepochs =   20;
opts.batchsize = 2;
opts.momentum  =   0;
opts.alpha     =   0.008;
%opts.plot = 1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
%figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rand('state',0)
%train dbn
dbn.sizes = [40 70];
opts.numepochs =   25;
opts.batchsize = 2;
opts.momentum  =   0;
opts.alpha     =   0.008;
opts.plot = 1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 1);
nn.activation_function = 'sigm';
nn.learningRate = 5;
nn.scaling_learningRate = 0.99;

%train nn
opts.numepochs =  500;
opts.batchsize = 20;
opts.plot = 1;
nn = nntrain(nn, train_x, train_y, opts);
%saveas(gcf,'nntrain.jpg');
[auc, predict] = nntest(nn, test_x, test_y);
%save nn
%assert(er < 0.10, 'Too big error');

end