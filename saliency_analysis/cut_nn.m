%% Remove the input units in index_cut from the NN while keep the other weights fixed 
function [reduced_nn] = cut_nn(nn,index_keep)
	%nn.size(1) = nn.size(1)-length(index_cut);
    index_cut = 1:86;
    index_cut(index_keep) = [];
    % notice the bias in the vision layer
	nn.W{1}(:,index_cut+1) = zeros([nn.size(2),length(index_cut)]);
	%nn.vW{1}(:,index_cut) = [];
	%nn.dW{1}(:,index_cut) = [];
	reduced_nn = nn;
end
