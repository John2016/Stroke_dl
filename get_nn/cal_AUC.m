%% LOGISTIC regression
function [nn, AUC] = cal_AUC(update_times)
% load('vessel_examin_original.mat')
load('vessel_update_1.mat')
load('vessel_delete_update.mat')
%[train_x, train_y, test_x, test_y] = preprocess(update_data(vessel_update_1));
[train_x, train_y, test_x, test_y] = preprocess_2(update_data(vessel_delete_update, update_times));
%[train_x, train_y, test_x, test_y] = preprocess_2(vessel_delete_update);

%  Setup and train a stacked denoising autoencoder (SDAE)
rand('state',0);
sae = saesetup([86 15 80]);
% 数据有负数，因此选tanh
sae.ae{1}.activation_function       =  'sigm';
%sae.ae{1}.learningRate              = 1.3;
sae.ae{1}.learningRate              = 0.0015;      
% 0.0005
%sae.ae{2}.learningRate              = 0.0005;          normal
sae.ae{2}.learningRate              = 0.0008; 
%sae.ae{1}.inputZeroMaskedFraction   = 0.1;
%opts.numepochs =   1;
opts.numepochs =   25;
opts.batchsize = 1;
sae = saetrain(sae, train_x, opts);
%visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
nn = nnsetup([86 15 80 1]);
nn.activation_function              = 'sigm';
%nn.learningRate                     = 1;
%nn.learningRate                     =  0.05;           single layer
nn.learningRate                     =  0.01;          %deleted_update data 
nn.weightPenaltyL2 = 1e-4;  %  L2 weight decay
nn.scaling_learningRate = 0.998;
%nn.dropoutFraction = 0.5;   %  Dropout fraction 
nn.output = 'sigm';
nn.W{1} = sae.ae{1}.W{1};
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
%opts.numepochs =   1;
%opts.numepochs =   800;
opts.numepochs = 700;
%nn.output = 'softmax';
opts.batchsize = 10;
opts.plot               = 1;                           %  enable plotting
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
%assert(er < 0.16, 'Too big error');
AUC = er;

