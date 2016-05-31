function [] = experiment_figure2(config, net)

category_lists(1).name = 'rose';

for ii = 1:length(category_lists)
    close all;
    gpuDevice();
    
    learningTime = tic;
    
    fprintf('Learning category: %s\n', category_lists(ii).name);
    if nargin < 2
        [config, net] = frame_config(category_lists(ii).name, 'dense', 0, 'codebook');
    end
    
    %% Step 1: add top layers
    for layer = 1:1:config.layer_to_learn
        img = randn(net.normalization.imageSize, 'single');
        net = vl_simplenn_move(net, 'gpu') ;
        
        if config.layer == 0
            switch config.learn_scheme
                case {'mle_e2e', 'hae_e2e', 'cd_e2e'}
                    net = add_bottom_filters(net, config);
                case {'mle_l2l', 'hae_l2l', 'cd_l2l'}
                    if layer < config.layer_to_learn
                        net = add_bottom_filters(net, config, layer);
                    end
            end
        end
        
        res = vl_simplenn(net, gpuArray(img));
        config.dydz_sz = size(res(end).x);
        
        %% Step 2: do some modifications on config
        
        res = vl_simplenn(net, gpuArray(img));
        net.numFilters = zeros(1, length(net.layers));
        for l = 1:length(net.layers)
            if isfield(net.layers{l}, 'weights')
                sz = size(res(l+1).x);
                net.numFilters(l) = sz(1) * sz(2);
            end
        end
        config.dydz_sz = size(res(end).x);
        if config.layer == 0
            config.layer_sets = numel(net.layers):-1:1;
        else
            config.layer_sets = numel(net.layers):-1:numel(net.layers);
            config.layer_sets = numel(net.layers):-1:1;
        end
        
        net = vl_simplenn_move(net, 'cpu') ;
        clear res;
        clear img;
        
         %% Step 3 create imdb
         imgCell = read_images(config, net);
         [imdb, getBatch] = convert2imdb(imgcell2mat(imgCell));
        
        %% Step 4: training
        switch config.learn_scheme
            case {'mle_l2l', 'mle_e2e'}
                net = train_model_generative(config, net, imdb, getBatch, layer);
                config.Gamma = config.Gamma / 2.5;
            case {'hae_l2l', 'hae_e2e'}
                net = train_model_hae(config, net, imdb, getBatch, layer);
            case {'cd_l2l', 'cd_e2e'}
                net = train_model_cd(config, net, imdb, getBatch, layer);
        end
    end
    learningTime = toc(learningTime);
    hrs = floor(learningTime / 3600);
    learningTime = mod(learningTime, 3600);
    mins = floor(learningTime / 60);
    secds = mod(learningTime, 60);
    fprintf('total learning time is %d hours / %d minutes / %.2f seconds.\n', hrs, mins, secds);
end