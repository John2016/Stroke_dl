%%求四个判断指标
function [ np_matrix ] = neg_pos_er( labels, expected )
%NEG_POS_ER Summary of this function goes here
%   Detailed explanation goes here
np_matrix = ones(2,2);
np_matrix(1,1) = sum(labels==1 & expected == 1);
np_matrix(1,2) = sum(labels==2 & expected == 1);
np_matrix(2,1) = sum(labels==1 & expected == 2);
np_matrix(2,2) = sum(labels==2 & expected == 2);
end

