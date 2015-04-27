%% compute and compare the top25, bottom25 and random25 based on the box-image
[ train_x, train_y, test_x, test_y ] = preprocess_3( vessel_delete_update );

%perform 
predict_y = nnpredict(nn_dbn, [train_x;test_x]);
index_high = find(predict_y>0.95&[train_y;test_y]==1);
%index_high = find(predict_y>0.999&[train_y;test_y]==1);
%index_high = index_high(4:2003);     %2171`
input_x = [train_x;test_x];
saliency_high_positive = saliency_cuzhong(nn_dbn, input_x(index_high,:));

% draw the box-image
figure;
boxplot(saliency_high_positive(:,index_colum_useful));

% sort by someway
