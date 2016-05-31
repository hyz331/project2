function net = add_bottom_filters(net, config, layer)

if nargin < 3
    layer = -1;
end

switch config.net_type
    case {'alexnet_5', 'alexnet_3'}
        if layer == -1
            layer = 1:str2double(config.net_type(9:end));
        end
        net = alexnet_net(net, layer);
    case 'vgg_16_3'
        net = vgg_16_3(net, config);
    case {'frame_3', 'frame_2', 'frame_1'}
        if layer == -1
            layer = 1:str2double(config.net_type(7:end));
        end
        net = frame_net(net, layer);
end

function net = vgg_16_3(net)

opts.scale = 1;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.weightInitMethod = 'gaussian' ;
opts.model = 'vgg_vd' ;
opts.batchNormalization = false;
opts.addrelu = true;

net = add_cnn_block(net, opts, '1_1', 3, 3, 3, 64, 1, 1) ;
net = add_cnn_block(net, opts, '1_2', 3, 3, 64, 64, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_cnn_block(net, opts, '2_1', 3, 3, 64, 128, 1, 1) ;
net = add_cnn_block(net, opts, '2_2', 3, 3, 128, 128, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_cnn_block(net, opts, '3_1', 3, 3, 128, 256, 1, 1) ;
net = add_cnn_block(net, opts, '3_2', 3, 3, 256, 256, 1, 1) ;
net = add_cnn_block(net, opts, '3_3', 3, 3, 256, 256, 1, 1) ;
% end



function net = alexnet_net(net, layer)

opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.weightInitMethod = 'gaussian' ;
opts.model = 'alexnet' ;
opts.batchNormalization = false;
opts.addrelu = true;

% successful setting for two layer end to end learning
% Gamma = 0.00001;
% layer_1
% num_out = 128
% filter_sz = 35;
% stride = 4;
% pad_sz = 4;

% layer_2
% num_out = 64
% filter_sz = 11;
% stride = 2;
% pad_sz = floor(filter_sz/2);

if layer >= 1
    %% layer 1
    layer_name = '1';
    num_in = 3;
    num_out = 96;
    filter_sz = 11; %11
    stride = 3; %2, 8
    pad_sz = 0;%floor(filter_sz/2);
    pad = ones(1,4)*pad_sz;
    net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad);
    net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
        'method', 'max', ...
        'pool', [2, 2], ...
        'stride', 2, ...
        'pad', 0) ;
end

if layer >= 2
    %% layer2
    layer_name = '2';
    num_in = 96;
    num_out = 64; % 64
    filter_sz = 7; %7
    stride = 2;%2
    pad_sz = 2;%ceil(filter_sz/2);
    pad = ones(1,4)*pad_sz;
    net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad) ;
    net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
        'method', 'max', ...
        'pool', [2, 2], ...
        'stride', 2, ...
        'pad', 0) ;
end

if layer >= 3
    %% layer3
    layer_name = '3';
    num_in = 128;
    num_out = 256;
    filter_sz = 3; %5
    stride = 1;%3
    pad_sz = ceil(filter_sz/2);
    pad = ones(1,4)*pad_sz;
    net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad);
end

if layer >= 4
    layer_name = '3';
    num_in = 256;
    num_out = 256;
    filter_sz = 3; %5
    stride = 1;%3
    pad_sz = 1;
    pad = ones(1,4)*pad_sz;
    net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad) ;
end

if layer >= 5
    layer_name = '5';
    num_in = 256;
    num_out = 128;
    filter_sz = 3; %5
    stride = 1;%3
    pad_sz = 1;
    pad = ones(1,4)*pad_sz;
    net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad) ;
end

function net = frame_net(net, layer)

opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.weightInitMethod = 'gaussian' ;
opts.model = 'alexnet' ;
opts.batchNormalization = false;
opts.addrelu = true;

if ismember(1, layer)
    %% layer 1
    layer_name = '1';
    num_in = 3;
    num_out = 100;
    filter_sz = 15; %11
    stride = 3; %2, 8, 3
    pad_sz = 2;%floor(filter_sz/2); %3
    pad = ones(1,4)*pad_sz;
    net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad);
%     net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
%         'method', 'max', ...
%         'pool', [2, 2], ...
%         'stride', 2, ...
%         'pad', 0) ;
end

if ismember(2, layer)
    %% layer2
    layer_name = '2';
    num_in = 100;
    num_out = 64; % 64 40
    filter_sz = 5; %7 size: 34, 5
    stride = 1;%2 , 1
    pad_sz = 2;%ceil(filter_sz/2);
    pad = ones(1,4)*pad_sz;
    net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad) ;
%     net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
%         'method', 'max', ...
%         'pool', [2, 2], ...
%         'stride', 2, ...
%         'pad', 0) ;
end
% 
if ismember(3, layer)
    %% layer3
    layer_name = '3';
    num_in = 64;
    num_out = 30; % 3
    filter_sz = 3; %5
    stride = 1;%3
    pad_sz = ceil(filter_sz/2);
    pad = ones(1,4)*pad_sz;
    net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad);
%     net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
%         'method', 'max', ...
%         'pool', [2, 2], ...
%         'stride', 2, ...
%         'pad', 0) ;
end
% 
% if ismember(4, layer)
%     %% layer3
%     layer_name = '3';
%     num_in = 32;
%     num_out = 16; % 3
%     filter_sz = 3; %5
%     stride = 1;%3
%     pad_sz = ceil(filter_sz/2);
%     pad = ones(1,4)*pad_sz;
%     net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad);
% %     net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
% %         'method', 'max', ...
% %         'pool', [2, 2], ...
% %         'stride', 2, ...
% %         'pad', 0) ;
% end

