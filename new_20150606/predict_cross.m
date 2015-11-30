function predict = predict_cross( cross_nn,input_x)
% PREDICT_CROSS Summary of this function goes here
%   Detailed explanation goes here
predict = zeros(size(input_x,1),1);
for ii = 1:10
    predict = predict + nnpredict(cross_nn(ii),input_x);
end
predict = predict / 10;

end

