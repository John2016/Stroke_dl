%% To analysis the Saliency_maps using the median-number, quarter-number
function [median,quarter_distance,signature] = quarter(saliency_map)
	quarter_points = prctile(saliency_map,[25,50,75]);

	median = quarter_points(2,:);
	quarter_distance = quarter_points(3,:) - quarter_points(1,:);
	signature = abs(quarter_points(3,:)) ./ quarter_distance - 0.5;
end
