function rHat = filter_response(net, imgs)
net = vl_simplenn_move(net, 'gpu') ;
if isa(imgs, 'gpuArray')
    res = vl_simplenn(net, imgs);
else
    res = vl_simplenn(net, gpuArray(imgs) );
end

rHat = [];
for i = max(1,length(net.layers)-2):length(net.layers)
    if isfield(net.layers{i}, 'weights')
        if ~isempty(res(i+1).x)
            tmp = mean(gather(res(i+1).x), 4);
        elseif ~isempty(res(i+2).x)
            tmp = mean(gather(res(i+2).x), 4);
        end
        rHat = [rHat, reshape(tmp, 1, [])];
    end
end
% rHat = reshape(rHat, [], size(rHat, 3));
% rHat = mean(rHat, 1);
% net = vl_simplenn_move(net, 'cpu') ;

