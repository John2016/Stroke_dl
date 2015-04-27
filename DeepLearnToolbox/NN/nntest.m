function [er, bad] = nntest(nn, x, y)
    labels = nnpredict(nn, x);
    [dummy, expected] = max(y,[],2);
    bad = find(labels ~= expected);    
    er = numel(bad) / size(x, 1);
    
    % update for bi-classification
    % @ John Lee
    % [auc, predictions]
   if nn.size(nn.n) == 1
        [tpr, fpr,thr] = roc(y', labels');
        %figure;
        %plotroc(y', labels');
        auc = trapz(fpr, tpr);
        er = auc;
        bad = labels;
   end
end
