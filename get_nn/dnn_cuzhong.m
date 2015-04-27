%%脑卒中
% remove inconsistent
load('vessel_examin_original.mat')
load('vessel_update_1.mat')
%[train_x, train_y, test_x, test_y] = preprocess(update_data(vessel_update_1));
[train_x, train_y, test_x, test_y] = preprocess(update_data(vessel_delete_update));
%[train_x, train_y, test_x, test_y] = preprocess(vessel_update_1);

%  Setup and train a stacked denoising autoencoder (SDAE)
rand('state',0);
sae = saesetup([86 15 50]);
% 数据有负数，因此选tanh
sae.ae{1}.activation_function       =  'sigm';
%sae.ae{1}.learningRate              = 1.3;
sae.ae{1}.learningRate              = 0.0015;
sae.ae{2}.learningRate              = 0.0005;
%sae.ae{1}.inputZeroMaskedFraction   = 0.1;
%opts.numepochs =   1;
opts.numepochs =   25;
opts.batchsize = 1;
sae = saetrain(sae, train_x, opts);
%visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
nn = nnsetup([86 15 50 2]);
nn.activation_function              = 'sigm';
%nn.learningRate                     = 1;
%nn.learningRate                     =  0.05;           single layer
nn.learningRate                     =  0.009;
nn.weightPenaltyL2 = 1e-4;  %  L2 weight decay
nn.dropoutFraction = 0.5;   %  Dropout fraction 
nn.output = 'sigm';
nn.W{1} = sae.ae{1}.W{1};
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
%opts.numepochs =   1;
opts.numepochs =   200;
%nn.output = 'softmax';
opts.batchsize = 10;
opts.plot               = 1;                           %  enable plotting
[er_p, bad_p] = nntest(nn, test_x, test_y);
nn = nntrain(nn, train_x, train_y, opts);
%nn = nntrain(nn, train_cuzhong_x, train_cuzhong_y, opts);
[er_tr, bad_tr] = nntest(nn, train_x, train_y);
labels = nnpredict(nn, test_x);
[dummy, expected] = max(test_y,[],2);
np_matrix = neg_pos_er(labels, expected);
bad = find(labels ~= expected);    
er = numel(bad) / size(test_x, 1);
%[er, bad] = nntest(nn, test_x, test_y);
er
assert(er < 0.16, 'Too big error');

