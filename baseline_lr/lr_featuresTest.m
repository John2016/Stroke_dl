%% Calculate and compare the top25 features of the logistic regression
index_top25 = index_absweight(1:25);
index_bottom25 = index_absweight(62:end);
index_random25 = randperm(86);
index_random25 = index_random25(11:35);
auc_lrtest = [];
nn_lrtest = [];
for index = [index_top25',index_bottom25',index_random25']
    input_data = vessel_delete_update(:,[1,index']);
    [nn_tmp,auc_tmp] = cal_dbn(input_data);
    nn_lrtest = [nn_lrtest, nn_tmp];
    auc_lrtest = [auc_lrtest, auc_tmp];
end

[tmp_weght,index_abs] = sort(abs(nn.W{1}(1:54)));

%% test 2: draw the curves of the 54 features
auc_cp_mat = [];
randlist = randperm(54);
for i = 1:54
    index_top = index_abs(1:i);
    index_bottom = index_abs(end-i+1:end);
    index_rand = randlist(1:i);
    index_original = 1:i;

    auc_compare = [];
    for index_test = [index_top',index_bottom',index_rand',index_original']
        [auc,pre] = nntest(cut_nn(nn,index_test'),[train_x;test_x],[train_y;test_y]);
        auc_compare = [auc_compare,auc];
    end
    auc_cp_mat = [auc_cp_mat;auc_compare];
end
