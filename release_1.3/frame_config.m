function [config, net_cpu] = frame_config(category, model_type, layer, exp_type, learn_scheme)
setenv('LD_LIBRARY_PATH', ['/usr/local/cuda/lib64:', getenv('LD_LIBRARY_PATH')]);
% we only support gpu
config.gpus = 1;

if nargin < 2
    model_type = 'sparse';
end

if nargin < 1
    category = 'cat';
end

if nargin < 3
    layer = 11;
end

if nargin < 4
    exp_type = 'object';
end

if nargin < 5
    config.learn_scheme = 'mle_l2l';
    % 'mle_l2l', 'mle_e2e';
    % 'cd_l2l', 'cd_e2e';
else
    config.learn_scheme = learn_scheme;
end

layer = max(0, layer);
config.layer = layer;


learningRate = 0.001 * ones(1, 37);
learningRate(34) = 0.8;
learningRate(30) = 0.1;
learningRate(29) = 0.1;
learningRate(24:28) = 0.005;
learningRate(15:23) = 0.001;
learningRate(10:14) = 0.0006;
learningRate(9) = 0.0002;
learningRate(7) = 0.0003;
learningRate(4) = 0.0001;
learningRate(2) = 0.0001;

config.sparse_level = ones(1, 37)*10;

% category name
config.categoryName = category;

% model type: can either be dense model or spase model
config.model_type = model_type;

% experiment type: currently supports 'object', 'texture' and 'codebook'
config.exp_type = exp_type;

% image path: where the dataset locates
if length(category) >= 6 && strcmpi(category(1:6), 'ILSVRC') == 1
    % imageNet
    imageNetPath = '../../data/';
    config.inPath = [imageNetPath, category];
    config.isImageNet = true;
    if ~exist(config.inPath, 'dir')
       fprintf('Error: No imageNet folder find: %s\n', config.inPath);
       keyboard;
    end
else
    % own dataset
    config.inPath = ['../Image/', exp_type, '/' config.categoryName '/'];
    config.isImageNet = false;
end 

% 3rd party path: where the matconvnn locates
config.matconvv_path = '../matconvnet-1.0-beta16/';

% model path: where the deep learning model locates
config.model_path = '../model/';

% deep learning model name: default we use vgg-16
% for more model, please visit
% http://www.vlfeat.org/matconvnet/pretrained/ for more information
config.model_name = 'imagenet-vgg-verydeep-16.mat';

% parameter for synthesis
% nTileRow \times nTileCol defines the number of paralle chains
% right now, we currently support square chains, e.g. 2*2, 6*6, 10*10 ...
config.nTileRow = 1; 
config.nTileCol = 1;

% standard deviation for reference model q(I/sigma^2)
% no need to change. 30 - 100 recommended
switch config.learn_scheme
    case {'cd_l2l', 'cd_e2e'}
        config.refsig = 1;
        config.Delta = 0.5; %0.0112 * 10; 0.5
    case {'mle_e2e', 'mle_l2l'}
        config.refsig = 1;
        config.Delta = 0.3; %0.0112 * 10; 0.5
end

% update rate for langevin dynamics
% no need to change.

config.warmGamma = 0.00001;



switch lower(config.exp_type)
    case 'object'
        if layer == 0
            config.net_type = 'frame_3';
            config.T = 30; % for warm start please use 100, for cold start use 200
            switch config.learn_scheme
                case {'mle_e2e', 'cd_e2e'}
                    config.layer_to_learn = 1;
                    config.nIteration = 3000; %30000;
                    config.Gamma = 0.0001;%0.0001;
                case {'mle_l2l', 'cd_l2l'}
                    config.layer_to_learn = 3;
                    config.nIteration = 600;
                    config.Gamma = 0.004;
            end   
        else
            config.layer_to_learn = 1;
            config.nIteration = 700; % dense frame could be 100 to 200, for warm start please use 150, for cold start use 200
            config.Gamma = 0.0000001;
            config.T = 30; % for warm start please use 100, for cold start use 200
        end
    case 'texture'
        if layer == 0
            config.net_type = 'frame_3';
            config.T = 10;
            switch config.learn_scheme
                case {'mle_e2e', 'cd_e2e'}
                    config.layer_to_learn = 1;
                    config.nIteration = 1200; %2000; 
                    config.Gamma = 0.0001; %0.00005;
                case {'mle_l2l', 'cd_l2l'}
                    config.layer_to_learn = 3;
                    config.nIteration = 700;
                    config.Gamma = 0.0005;
            end   
        else
            config.layer_to_learn = 1;
            config.nIteration = 1000;
            config.Gamma = 0.000001;
            config.T = 10;
        end
    case 'codebook'
        if layer == 0
            config.T = 20;
            config.net_type = 'frame_3';
            config.top_type = 'f_1';
            switch config.learn_scheme
                case {'mle_e2e', 'cd_e2e'}
                    config.layer_to_learn = 1;
                    config.nIteration = 2000; %2000; 
                    config.Gamma = 0.0001; %0.00005; 0.0001 dense uses 0.0003; sparse uses
                case {'mle_l2l', 'cd_l2l'}
                    config.layer_to_learn = 3;
                    config.nIteration = 700;
                    config.Gamma = 0.001;
            end           
        else
            config.layer_to_learn = 1;
            config.nIteration = 3000;
            config.T = 50;
            config.top_type = 'f_1';
            switch config.top_type
                case 'f_vgg_1'
                    config.Gamma = learningRate(layer)/30000*10;
                    config.Delta = 0.0112*3;
                case 'f_1'
                    config.Gamma = learningRate(layer)/3000*30; %learningRate(layer)/30000*30;
                    config.Delta = 0.0112 * 3;
            end
        end
    otherwise
end

config.forceLearn = true;
config.BatchSize = 32;
config.issave = false;

run(fullfile(config.matconvv_path, 'matlab', 'vl_setupnn.m'));

net_cpu = load([config.model_path, config.model_name]);
net_cpu = net_cpu.net;

% net_cpu.normalization.imageSize = [100, 100, 3];
% net_cpu.normalization.averageImage = imresize(net_cpu.normalization.averageImage,net_cpu.normalization.imageSize(1:2));

config.sx = net_cpu.normalization.imageSize(1);
config.sy = net_cpu.normalization.imageSize(2);
if layer == 0
    net_cpu.layers = {};
else
    net_cpu.layers = net_cpu.layers(1:layer);
end

% result file: no need to change
if layer == 0
    config.working_folder = ['./working/', exp_type, '/', config.categoryName, '_', config.model_type, '_net_', config.net_type, '_', config.learn_scheme, '/'];
    config.Synfolder = ['./synthesiedImage/', exp_type, '/', config.categoryName, '_', config.model_type, '_net_', config.net_type, '_', config.learn_scheme, '/'];
    config.figure_folder = ['./figure/', exp_type, '/', config.categoryName, '_', config.model_type, '_net_', config.net_type, '_', config.learn_scheme, '/'];
else
    config.working_folder = ['./working/', exp_type, '/', config.categoryName, '_', config.model_type, '_layer_', num2str(layer), '_', config.learn_scheme, '/'];
    config.Synfolder = ['./synthesiedImage/', exp_type, '/', config.categoryName, '_', config.model_type, '_layer_', num2str(layer), '_', config.learn_scheme, '/'];
    config.figure_folder = ['./figure/', exp_type, '/', config.categoryName, '_', config.model_type, '_layer_', num2str(layer), '_', config.learn_scheme, '/'];
end

% create directory
if ~exist('./working/', 'dir')
    mkdir('./working/')
end

if ~exist(['./working/', exp_type], 'dir')
   mkdir(['./working/', exp_type]) 
end

if ~exist('./synthesiedImage/', 'dir')
   mkdir('./synthesiedImage/') 
end

if ~exist(['./synthesiedImage/', exp_type], 'dir')
   mkdir(['./synthesiedImage/', exp_type]) 
end

if ~exist('./figure/', 'dir')
   mkdir('./figure/') 
end

if ~exist(['./figure/', exp_type], 'dir')
   mkdir(['./figure/', exp_type]) 
end

if ~exist(config.Synfolder, 'dir')
   mkdir(config.Synfolder);
end

if ~exist(config.working_folder, 'dir')
    mkdir(config.working_folder);
end

if ~exist(config.figure_folder, 'dir')
    mkdir(config.figure_folder);
end

