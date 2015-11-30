function auc = test_cross( nn_cross, xx, yy )
%   TEST_CROSS: test the AUC and predicted value
%   Detailed explanation goes here
    labels = predict_cross(nn_cross, xx);

    % update for bi-classification
    % @ John Lee
    % [auc, predictions]
    [tpr, fpr,thr] = roc(yy', labels');
    %figure;
    %plotroc(y', labels');
    auc = trapz(fpr, tpr);
    er = auc;
    %bad = labels;
end

