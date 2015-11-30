%% train a deep NN using SAE model 
function [nn, AUC] = train_sae(train_x, train_y, test_x, test_y)
%%  Setup and train a stacked denoising autoencoder (SDAE)
rand('state',0);
sae = saesetup([261 150 70 35])

sae.ae{1}.activation_function       =  'sigm';
%sae.ae{1}.learningRate              = 0.0022; 
sae.ae{1}.learningRate              = 0.0025;     
%sae.ae{2}.learningRate           = 0.0045; 
sae.ae{2}.learningRate              = 0.005; 
%sae.ae{3}.learningRate              = 0.0020
sae.ae{3}.learningRate              = 0.0025;
%sae.ae{1}.inputZeroMaskedFraction   = 0.08;
%sae.ae{2}.inputZeroMaskedFraction   = 0.08;
%sae.ae{3}.inputZeroMaskedFraction   = 0.05;

%opts.numepochs =   25;
opts.numepochs =   30;
%opts.batchsize = 10;
opts.batchsize = 20;
opts.plot = 1;

sae = saetrain(sae, train_x, opts);
%visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
nn = nnsetup([261 150 70 35 2]);
nn.activation_function              = 'sigm';
%nn.learningRate                     =  0.20; 
nn.learningRate                     =  0.25; 
% nn.weightPenaltyL2 = 1e-4;  %  L2 weight decay
nn.weightPenaltyL2 = 1e-4;  %  L2 weight decay
%nn.scaling_learningRate = 0.9985;
nn.scaling_learningRate = 0.9984;
% nn.dropoutFraction = 0.05;
nn.dropoutFraction = 0.1;   %  Dropout fraction 
nn.output = 'softmax';
nn.W{1} = sae.ae{1}.W{1};
nn.W{2} = sae.ae{2}.W{1};
nn.W{3} = sae.ae{3}.W{1};
%nn.inputZeroMaskedFraction   = 0.05;

% Train the FFNN
%opts.numepochs = 1500;
opts.numepochs = 3000;
%opts.batchsize = 50;
opts.batchsize = 100;
opts.plot               = 1;                           %  enable plotting
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
%assert(er < 0.16, 'Too big error');
AUC = er;

