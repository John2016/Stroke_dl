%%��ԭ���ݼ��д���NA��99ȫ���滻Ϊ0
function [ vessel_update_1] = delete_na( data )
%DELETE_NA Summary of this function goes here
%   Detailed explanation goes here
vessel_update_1 = data(:, 2:end);
vessel_update_1(find(vessel_update_1(:,:)==99)) = 0;
vessel_update_1 = [data(:, 1), vessel_update_1];

end

