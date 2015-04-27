%% using the quarter data of the saliency_maps to analysis those features
saliency_maps = saliency_high_positive;
[median,q_distance,q_sig] = quarter(saliency_maps);

figure; bar(median); colormap(spring);
title('Median number of the 86 features according to the Saliency Maps');
saveas(gcf,'Median-SaliencyMaps.jpg');

figure; bar(q_distance); colormap(summer);
title('Quarters-Distance of the 86 features according to the Saliency Maps');
saveas(gcf,'Quarters-Distance.jpg');

figure; bar(q_sig); colormap(autumn);
title('Signature of the 86 features according to the Saliency Maps');
saveas(gcf,'Signature-SaliencyMaps.jpg');

% test 1: normal
[sorted_dis,index] = sort(q_distance);
index_top25 = index(1:25);
index_bottom25 = index(end-24:end);
index_random25 = randi([1 86],1,25);

% test 2: the left 54
auc_cp_mat = [];
for i = [10,20,25,30,40,45,50,54]
    [sorted_dis,index] = sort(q_distance(1:54));
    index_top25 = index(1:i);
    index_bottom25 = index(end-i+1:end);
    index_rand = randperm(54);
    index_random25 = index_rand(1:i);

    auc_compare = [];
    for index_test = [index_top25',index_random25',index_bottom25']
        [auc,pre] = nntest(cut_nn(nn_dbn,index_test'),test_x,test_y);
        auc_compare = [auc_compare,auc];
    end
    auc_cp_mat = [auc_cp_mat;auc_compare];
end

% test 3: draw tow curves: top-down and bottom-up according to saliency maps
auc_cp_mat = [];
for i = 1:54
    [sorted_dis,index] = sort(q_distance(1:54));
    index_top25 = index(1:i);
    index_bottom25 = index(end-i+1:end);

    auc_compare = [];
    for index_test = [index_top25',index_bottom25']
        [auc,pre] = nntest(cut_nn(nn_dbn,index_test'),test_x,test_y);
        auc_compare = [auc_compare,auc];
    end
    auc_cp_mat = [auc_cp_mat;auc_compare];
end

% test 4: draw two curves: in original order and in random order
auc_cp_mat = [];
index_rand = randperm(54);
for i = 1:54
    index_left = 1:i;
    index_random = index_rand(1:i);
    
    auc_compare = [];
    for index_test = [index_left',index_random']
        [auc,pre] = nntest(cut_nn(nn_dbn,index_test'),test_x,test_y);
        auc_compare = [auc_compare,auc];
    end
    auc_cp_mat = [auc_cp_mat;auc_compare];
end


% 1:54
new_nn = cut_nn(nn_dbn,1:54);
[auc,pre] = nntest(new_nn,test_x,test_y);


