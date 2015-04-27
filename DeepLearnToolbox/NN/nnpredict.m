function labels = nnpredict(nn, x)
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;
    
    [dummy, i] = max(nn.a{end},[],2);
    labels = i;
    
    % update for bi-classification 2015-3-26
    % @ John Lee
    if nn.size(nn.n) == 1
        labels = dummy;
    end
end
