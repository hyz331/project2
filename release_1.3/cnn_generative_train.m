function [net,  opts, train, config] = cnn_generative_train(net, imdb, syn_mat, getBatch, config, varargin)
% CNN_TRAIN   Demonstrates training a CNN
%    CNN_TRAIN() is an example learner implementing stochastlayer_sets = numel(net.layers):-1:numel(net.layers)-2;ic
%    gradient descent with momentum to train a CNN. It can be used
%    with different datasets and tasks by providing a suitable
%    getBatch function.
%
%    The function automatically restarts after each training epoch by
%    checkpointing.
%
%    The function supports training on CPU or on one or more GPUs
%    (specify the list of GPU IDs in the `gpus` option). Multi-GPU
%    support is relatively primitive but sufficient to obtain a
%    noticable speedup.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.numEpochs = 300 ;
opts.gpus = [] ; % which GPU devices to use (none, one, or more)
opts.learningRate = 0.001 ;
opts.continue = false ;
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.prefetch = false ;
opts.cudnn = true ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.errorFunction = 'multiclass' ;
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts = vl_argparse(opts, varargin) ;

% if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

evaluateMode = isempty(opts.train) ;

if ~evaluateMode
    for i=1:numel(net.layers)
        if isfield(net.layers{i}, 'weights')
            J = numel(net.layers{i}.weights) ;
            for j=1:J
                net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
            end
            if ~isfield(net.layers{i}, 'learningRate')
                net.layers{i}.learningRate = ones(1, J, 'single') ;
            end
            if ~isfield(net.layers{i}, 'weightDecay')
                net.layers{i}.weightDecay = ones(1, J, 'single') ;
            end
        end
    end
end

% setup GPUs
% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------
syn_mat = gpuArray(syn_mat);
for epoch=1:opts.numEpochs
    learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
    
    % train one epoch and validate
    train = opts.train(randperm(numel(opts.train))) ; % shuffle
    
    [net,stats.train] = process_epoch(opts, getBatch, epoch, train, learningRate, imdb, net, syn_mat, config) ;
end

% end of the function

% -------------------------------------------------------------------------
function  [net_cpu,stats] = process_epoch(opts, getBatch, epoch, subset, learningRate, imdb, net_cpu, syn_mat, config)
% -------------------------------------------------------------------------

% move CNN to GPU as needed
% numGpus = numel(opts.gpus) ;
% if numGpus >= 1
net = vl_simplenn_move(net_cpu, 'gpu') ;
% else
%   net = net_cpu ;
%   net_cpu = [] ;
% end

% validation mode if learning rate is zero
training = learningRate > 0 ;
if training, mode = 'training' ; else, mode = 'validation' ; end
if nargout > 2, mpiprofile on ; end

numGpus = numel(opts.gpus) ;

dydz_syn = gpuArray(zeros(config.dydz_sz, 'single'));
dydz_syn(net.filterSelected) = net.selectedLambdas;
dydz_syn = repmat(dydz_syn, 1, 1, 1, size(syn_mat, 4));
% dydz_syn = gpuArray( -ones([config.dydz_sz, size(syn_mat, 4)], 'single'));

for t=1:opts.batchSize:numel(subset)
    fprintf('%s: epoch %02d: batch %3d/%3d: ', mode, epoch, ...
        fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
    batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
    batchTime = tic ;
    numDone = 0 ;
    res = [] ;
    res_syn = [];
    stats = [] ;
    %   error = [] ;
    for s=1:opts.numSubBatches
        % get this image batch and prefetch the next
        batchStart = t + (labindex-1) + (s-1) * numlabs ;
        batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
        im = getBatch(imdb, batch) ;
        
        if opts.prefetch
            if s==opts.numSubBatches
                batchStart = t + (labindex-1) + opts.batchSize ;
                batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
            else
                batchStart = batchStart + numlabs ;
            end
            nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
            getBatch(imdb, nextBatch) ;
        end
        
        if numGpus >= 1
            im = gpuArray(im) ;
        end
        
        % training images
        numImages = size(im, 4);
        
        dydz = gpuArray(zeros(config.dydz_sz, 'single'));
        dydz(net.filterSelected) = net.selectedLambdas;
        dydz = repmat(dydz, 1, 1, 1, numImages);
        
        switch config.learn_scheme
            case {'mle_l2l', 'mle_e2e'}
                res = vl_simplenn(net, im, dydz, res, ...
                    'accumulate', s ~= 1, ...
                    'disableDropout', ~training, ...
                    'conserveMemory', opts.conserveMemory, ...
                    'backPropDepth', opts.backPropDepth, ...
                    'sync', opts.sync, ...
                    'cudnn', opts.cudnn) ;
                
                res_syn = vl_simplenn(net, syn_mat, dydz_syn, res_syn, ...
                    'accumulate', s ~= 1, ...
                    'disableDropout', ~training, ...
                    'conserveMemory', opts.conserveMemory, ...
                    'backPropDepth', opts.backPropDepth, ...
                    'sync', opts.sync, ...
                    'cudnn', opts.cudnn) ;
            case {'hae_l2l', 'hae_e2e'}

                [~, indicators] = vl_hae(net, im, dydz, res, [], ...
                    'accumulate', s ~= 1, ...
                    'disableDropout', ~training, ...
                    'conserveMemory', opts.conserveMemory, ...
                    'backPropDepth', opts.backPropDepth, ...
                    'sync', opts.sync, ...
                    'cudnn', opts.cudnn) ;
                
                res = vl_simplenn(net, im, dydz, [], 'conserveMemory', 1, 'cudnn', 1);
                syn_mat = (im- config.refsig^2*res(1).dzdx);
                res = [];
                
                res = vl_hae(net, syn_mat, dydz, res, indicators, ...
                    'accumulate', s ~= 1, ...
                    'disableDropout', ~training, ...
                    'conserveMemory', opts.conserveMemory, ...
                    'backPropDepth', opts.backPropDepth, ...
                    'sync', opts.sync, ...
                    'cudnn', opts.cudnn) ;
                clear indicators;
        end
        numDone = numDone + numel(batch) ;
    end
    
    % gather and accumulate gradients across labs
    if training
        [net,~] = accumulate_gradients(opts, learningRate, batchSize, size(syn_mat, 4), net, res, res_syn, config) ;
    end
    
    clear res;
    clear res_syn;
    
    % print learning statistics
    batchTime = toc(batchTime) ;
    %   stats = sum([stats,[batchTime ; error]],2); % works even when stats=[]
    speed = batchSize/batchTime ;
    
    fprintf(' %.2f s (%.1f data/s)', batchTime, speed) ;
    fprintf(' [%d/%d]', numDone, batchSize);
    fprintf('\n') ;
end

net_cpu = vl_simplenn_move(net, 'cpu') ;


% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(opts, lr, batchSize, num_syn, net, res, res_syn, config)
% -------------------------------------------------------------------------
layer_sets = config.layer_sets;
% if nargin < 8
%     layer_sets = numel(net.layers):-1:numel(net.layers)-2;
% end

for l = layer_sets
    for j=1:numel(res(l).dzdw)
        thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
        thisLR = lr * net.layers{l}.learningRate(j) ;
        
        if isfield(net.layers{l}, 'weights')
            %         net.layers{l}.momentum{j} = ...
            %             opts.momentum * net.layers{l}.momentum{j} ...
            %             - thisDecay * net.layers{l}.weights{j} ...
            %             + thisLR * (1 / batchSize) * res(l).dzdw{j} ...
            %             - thisLR * (1 / num_syn) * res_syn(l).dzdw{j} ;
            
            
            
            switch config.learn_scheme
                case {'mle_l2l', 'mle_e2e'}
                    gradient_dzdw = ((1 / batchSize) * res(l).dzdw{j} -  ...
                        (1 / num_syn) * res_syn(l).dzdw{j}) / net.numFilters(l);
                    if max(gradient_dzdw(:)) > 100 %10
                        gradient_dzdw = gradient_dzdw / max(gradient_dzdw(:)) * 100;
                    end
                case {'hae_l2l', 'hae_e2e'}
                    gradient_dzdw = config.refsig^2*((1 / batchSize) * res(l).dzdw{j}) / net.numFilters(l);
                    if max(gradient_dzdw(:)) > 100 %102
                        gradient_dzdw = gradient_dzdw / max(gradient_dzdw(:)) * 100;
                    end
            end
            
            net.layers{l}.momentum{j} = ...
                + opts.momentum * net.layers{l}.momentum{j} ...
                - thisDecay * net.layers{l}.weights{j} ...
                + gradient_dzdw;
            
            %             net.layers{l}.momentum{j} = gradient_dzdw;
            net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR *net.layers{l}.momentum{j};
            
            if j == 1
                res_l = min(l+2, length(res));
                fprintf('\n layer %s:max response is %f, min response is %f.\n', net.layers{l}.name, max(res(res_l).x(:)), min(res(res_l).x(:)));
                fprintf('max gradient is %f, min gradient is %f, learning rate is %f\n', max(gradient_dzdw(:)), min(gradient_dzdw(:)), thisLR);
            end
            %       net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR * gradient_dzdw;
        end
    end
end
