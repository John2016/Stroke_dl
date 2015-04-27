%%在选定数据集和指定nn上跑saliency maps算法
function [ saliency_map ] = saliency_cuzhong( nn, input_xs )
%SALIENCY_CUZHONG Summary of this function goes here
%   Detailed explanation goes here
%feed forward 
% 由于并不计算batch的err，因此batch-size可以随意设置，所以就一次计算吧
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

    n = nn.n;
    [nrow, ncol] = size(input_xs);
    saliency_map = ones([nrow ncol ]);
    
    for j = 1 : nrow
        input_x = input_xs(j,:);
        input_x = [1 input_x];
        nn.a{1} = input_x;
        nn.saliency = eye(ncol); 
        
        % perform a feedforward pass
        for i = 2 : n-1
            nn.linear{i} = nn.a{i - 1} * nn.W{i - 1}';
            switch nn.activation_function 
                case 'sigm'
                    % Calculate the unit's outputs (including the bias term)
                    nn.a{i} = sigm(nn.linear{i});
                case 'tanh_opt'
                    nn.a{i} = tanh_opt(nn.linear{i});
            end
             %Add the bias term
            nn.a{i} = [ones(1,1) nn.a{i}];
            %'sigm' specifically 
            %nn.saliency = nn.saliency * sigm(nn.linear{i}) .* (1-sigm(nn.linear{i})) .* nn.W{i-1}';
            size(sigm(nn.linear{i}));
            size(repmat(sigm(nn.linear{i}) .* (1-sigm(nn.linear{i})), nn.size(i-1), 1));
            size( nn.W{i-1}');
            nn.saliency = nn.saliency * (repmat(sigm(nn.linear{i}) .* (1-sigm(nn.linear{i})), nn.size(i-1), 1) .* nn.W{i-1}(:,2:end)');
        end

        nn.linear{n} = nn.a{n - 1} * nn.W{n - 1}';
        switch nn.output
            case 'sigm'
                nn.a{n} = sigm(nn.linear{n});
            case 'linear'
                nn.a{n} = nn.linear{n};
            case 'softmax'
                nn.a{n} = nn.linear{n};
                nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
                nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2)); 
        end
        %'sigm' specifically
        nn.saliency = nn.saliency * (repmat(sigm(nn.linear{n}),nn.size(n-1),1) .* (1-sigm(nn.linear{n})) .* nn.W{n-1}(:,2:end)');
        
        saliency_map(j,:) = nn.saliency;
    end
end





