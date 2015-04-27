%% Get the saliency maps for the specified NN

%% only the positive data
[ train_x, train_y, test_x, test_y ] = preprocess_2( vessel_delete_update );
positive_x = [train_x; test_x];
positive_y = [train_y; test_y];
positive_x = positive_x(positive_y==1,:);

%perform 
saliency_positive = saliency_cuzhong(result_matrix(2).nn, positive_x);
predict_y = nnpredict(result_matrix(2).nn, positive_x);

% view the distribution of the saliency on positive data
p = ones([1 86]);
figure;
title('The Distribution of the saliency_maps on every colomuns')
for ii = 1:9
    for jj = 1:10
        number = 10 * (ii-1)+jj;
        if number == 87
            break
        end
        subplot(10,10, number)
        p(number) = capaplot(saliency_positive(:,number),[-2,2]);
        colormap(cool);
        title(sprintf('column %d',number));
    end
end

% histfit()
for ii = 1: 5
    for jj = 1:5
        number = (ii-1)*5 + jj;
        subplot(5,5,number);
        histfit(saliency_positive(:,number))
        title(sprintf('%d',number))
    end
end

%analysis
%get the average value for 86 cols
average_o = mean(saliency_positive);
average_abs = mean(abs(saliency_positive));
%get the average value of the top 1500 prediction
maps_up_81 = saliency_positive(predict_y>0.81,:);
average_top_1500 = mean(maps_up_81(1:1500,:));
average_top1500_abs = mean(abs(maps_up_81(1:1500,:)));
%get the 45 rows which's prediction > 0.99
saliency_up_99 = saliency_positive(predict_y>0.99,:);
average_top_45 = mean(saliency_up_99);
average_top45_abs = mean(abs(saliency_up_99));

%plot
subplot(2,2,1)
bar(average_o)
title('Figure 1: Average Positive')

subplot(2,2,2)
bar(average_top_1500)
title('Figure 3: Average Top_1500')

subplot(2,2,3)
bar(average_top_45)
title('Figure 4: Average Top_45')

average_saliency = [average_o; average_top_1500; average_top_45];
subplot(2,2,4)
imagesc(average_saliency)
colormap(cool)
colorbar

%plot another
figure
subplot(2,2,1)
bar(average_abs)
title('Figure 1: Average Positive')

subplot(2,2,2)
bar(average_top1500_abs)
title('Figure 3: Average Top_1500')

subplot(2,2,3)
bar(average_top45_abs)
title('Figure 4: Average Top_45')

average_abs_saliency = [average_abs; average_top1500_abs; average_top45_abs];
subplot(2,2,4)
imagesc(average_abs_saliency)
colormap(cool)
colorbar

%get the top-10 features
[sorted_features, index] = sort(abs(average_o));
[sorted_features_abs, index_abs] = sort(average_abs);