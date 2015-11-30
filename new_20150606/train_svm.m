function [ svm, er ] = train_svm( train_x, train_y, test_x, test_y )
%TRAIN_SVM Summary of this function goes here
%   Detailed explanation goes here
    options.MaxIter = 100000;
    svm = svmtrain(train_x, train_y,'Kernel_Function','rbf','Options',options);

    labels = svmclassify(svm, test_x);

    %[tpr, fpr,thr] = roc(test_y', labels');
    %figure;
    %plotroc(test_y', labels');
    %auc = trapz(fpr, tpr);
    er = sum(labels~=test_y)/size(test_y,1);

% start with Kernel_Function set to 'rbf' and default parameters
% try different parameters for training, and check via cross validation to
% obtain the best parameters
% the most important parameters to try changing are 'boxconstraint' and 'rbf_sigma'

% alternatively, optimizing your parameters with fminsearch

end

