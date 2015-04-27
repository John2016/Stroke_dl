%% To test whether the ratio of positive data affect the result_NN
result_matrix = [];
tic;
for ii = 1: 10: 101
    [nn, auc] = cal_AUC(ii);
    saveas(gcf,sprintf('%d.png',ii));
    close(gcf);
    trained.times     = ii;
    trained.nn           = nn;
    trained.auc         = auc;
    result_matrix   =  [result_matrix, trained];
    save result_matrix;
end
toc

% 统计出每一个times对应的auc并绘图
auc_each = ones([1,11]);
for ii = 1:11
    auc_each(ii) = result_matrix(ii).auc;
end
plot(1:11,auc_each),title('Figure 1: AUCs of their own datasets for each exp');

%计算12个结果与阳性数据比例的关系
 tmp_rate = ones([1,11]); 
for ii = 1:11
     jj = 1 + 10 * (ii-1);
     tmp_rate(ii) = 0.0730*jj/(1-0.0730+0.0730*jj);
end
figure;
plot(tmp_rate,auc_each),title('Figure 2: AUCs to positive-data rate');

%计算11个nn在vessel_delete_update数据集上的表现
auc_delete_update = ones([1,11]);
for ii = 1:11
    [train_x, train_y, test_x, test_y] = preprocess_2(vessel_delete_update);
    [auc, labels] = nntest(result_matrix(ii).nn,[train_x;test_x],[train_y;test_y]);
    auc_delete_update(ii) = auc;
end

%计算11个nn在未删除冲突数据的数据集上的表现
auc_update_1 = ones([1,11]);
for ii = 1:11
    [train_x, train_y, test_x, test_y] = preprocess_2(vessel_update_1);
    [auc, labels] = nntest(result_matrix(ii).nn,[train_x;test_x],[train_y;test_y]);
    auc_update_1(ii) = auc;
end

result_exp_2 = [[1:11]' tmp_rate' auc_each' auc_delete_update' auc_update_1' ];
save result_exp_2;

