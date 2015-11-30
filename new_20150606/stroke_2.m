function [ result_dbn] = stroke_2( stroke_data )
%STROKE_MAIN is the main function of the new process
%   Detailed explanation goes here
stroke_data=stroke_data(randperm(size(stroke_data,1)),:);
%% 10-fold cross-validation
% output: test_set && train_set
result_dbn = cell(1,10);
record_auc = zeros(1,10);
%indices = crossvalind('Kfold',size(stroke_data(:,1)),10);
indices = crossvalind('Kfold',size(stroke_data,1),5);
%disp(indices(1:100));
%for ii = 1:10
parfor ii = 1:2
    %disp(ii);
    %test_index = (indices == ii);
    %train_index = ~test_index;
    test_set = stroke_data(indices==ii,:);
    %disp(size(test_set));
    train_set = stroke_data(indices~=ii,:);
    %disp(size(train_set));
    % balance the positive and negtive examples in train_set
    % output: train_set
    %sum(train_set(:,1) == 0)
    %sum(train_set(:,1) == 1)
    times = sum(train_set(:,2) == 0) / sum(train_set(:,2) == 1);
    disp(sprintf('times:%d',times));
    %disp(sprintf('original: positve%d; negtive:%d',sum(train_set(:,2)==1),sum(train_set(:,2)==0)));
    train_positive = train_set(train_set(:,2) == 1,:);
    %disp(sum(train_positive(:,1)==0));
    %[nrow, ncol] = size(train_positive);
    for jj = 1:floor(times)
        train_set = [train_set;train_positive];
        %disp(sprintf('%d %d',sum(train_set(:,1)==0),sum(train_set(:,1)==1)));
    end

    % add the rest
    %nrest = floor((times - floor(times)) * nrow);
    %disp(nrest);
    %rest_index = randperm(nrow);
    %disp(rest_index(1:10));
    %rest_index = rest_index(1:nrest);
    %train_set = [train_set;train_positive(rest_index,:)];
    
    %% 100 * integer times
    [nrow,ncol] = size(train_set);
    train_set = train_set(1 : 100 * floor(nrow/100),:);
    train_set = train_set(randperm(size(train_set,1)),:);
    %disp(sum(train_set(:,1)==1));
    %disp(sum(train_set(:,1)==0));
   
    %% separate
    train_x = train_set(:,3:end);
    train_y = train_set(:,1:2);
    test_x = test_set(:,3:end);
    test_y = test_set(:,1:2);

    %% train and test
    % output: the deep model and AUC
    disp(sprintf('positive:%d, negtive:%d',sum(train_y(:,1)==0),sum(train_y(:,1)==1)));
    [nn, er] = train_dbn_sig(train_x, train_y, test_x, test_y);
    %saveas(gcf,sprintf('%d_train_bin_sae.png',ii));
    %disp(acc);
    result_dbn{ii} = nn;
    record_auc(ii) = er;
end
%save('result_dbn.mat','result_dbn');
%save('auc_dbn_cross.mat','record_auc');
record_auc
%disp(sum(record_auc/10));
end

