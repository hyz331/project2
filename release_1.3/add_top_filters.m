function net = add_top_filters(net, config)
switch net.layers{end}.type
    case 'pool'
        in = size(net.layers{end-2}.weights{1}, 4);
    case 'relu'
        in = size(net.layers{end-1}.weights{1}, 4);
    case {'fc', 'conv'}
        in = size(net.layers{end}.weights{1}, 4);
    otherwise
        fprintf('Not supported layer');
        keyboard;
end

switch config.exp_type
    case 'object'
        net = net_object(net, in, config.dydz_sz(1));
    case 'texture'
        net = net_texture(net, in);
    case 'codebook'
    switch config.top_type
        case 'f_1'
            net = net_f_1(net, in);
        case 'f_vgg_1'
            net = net_f_vgg_1(net, in);
        otherwise
            fprintf('Not supported type');
            keyboard;
    end
end
end

function net = net_object(net, in, filter_sz)
numFilters = 1; %% 
stride = 1;
pad_sz = 0;
pad = ones(1,4)*pad_sz;

opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.weightInitMethod = 'gaussian' ;
opts.model = 'alexnet' ;
opts.batchNormalization = false;
opts.addrelu = false;

layer_name = '4_1';

net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, in, numFilters, stride, pad);
end

function net = net_texture(net, in)
numFilters = 10; %% 
stride = 1;
pad_sz = 0;
filter_sz = 1;
pad = ones(1,4)*pad_sz;

opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.weightInitMethod = 'gaussian' ;
opts.model = 'alexnet' ;
opts.batchNormalization = false;
opts.addrelu = false;

layer_name = '4_1';

net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, in, numFilters, stride, pad);
end

function net = net_f_1(net, in, filter_sz)
if nargin < 3
    filter_sz = 7; %% 11
end
numFilters = 20; %% 
stride = 3; % 3
pad_sz = 4; % 4
pad = ones(1,4)*pad_sz;

opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.weightInitMethod = 'gaussian' ;
opts.model = 'alexnet' ;
opts.batchNormalization = false;
opts.addrelu = true;

layer_name = '4_1';

net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, in, numFilters, stride, pad);

% net.layers{end+1} = struct('type', 'pool', 'name', 'pool4', ...
%                            'method', 'max', ...
%                            'pool', [2, 2], ...
%                            'stride', 2, ...
%                            'pad', 0) ;
end

function net = net_f_vgg_1(net, in)
opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.weightInitMethod = 'gaussian' ;
opts.model = 'alexnet' ;
opts.batchNormalization = false;
opts.addrelu = true;

net = add_cnn_block(net, opts, '4_1', 5, 5, in, 64, 2, 1) ;
net = add_cnn_block(net, opts, '4_2', 5, 5, 64, 32, 2, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool4', ...
                           'method', 'max', ...
                           'pool', [3, 3], ...
                           'stride', 3, ...
                           'pad', 0) ;
end


