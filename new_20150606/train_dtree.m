function [dtree,er] = train_dtree(train_x,train_y,test_x,test_y)
    dtree = classregtree(train_x,train_y,'method','classification');
    labels = eval(dtree,test_x);
    labels = str2double(labels);
    er = sum(labels ~= test_y) / size(labels,1);
end
