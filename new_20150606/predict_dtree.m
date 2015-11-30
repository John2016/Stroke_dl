function dtree_labels = predict_dtree(my_dtree,test_x)
    %% get the predicted value based on the 10 regtree
    labels = [];
    for ii = 1:10
	labels = [labels,str2double(eval(my_dtree{ii},test_x))];
    end
    labels = round(sum(labels,2)/10);
    dtree_labels = labels;

end
