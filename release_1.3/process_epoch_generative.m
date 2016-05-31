function  [net_cpu, syn_mats] = process_epoch_generative(opts, getBatch, epoch, subset, learningRate, imdb, net_cpu, syn_mats, config)
% -------------------------------------------------------------------------

% move CNN to GPU as needed
numGpus = numel(opts.gpus) ;
net = vl_simplenn_move(net_cpu, 'gpu') ;

% validation mode if learning rate is zero
training = learningRate > 0 ;
if training, mode = 'training' ; else, mode = 'validation' ; end
if nargout > 2, mpiprofile on ; end

mmap = [] ;
mmap_bias = [];

dydz_syn = gpuArray(zeros(config.dydz_sz, 'single'));
dydz_syn(net.filterSelected) = net.selectedLambdas;
dydz_syn = repmat(dydz_syn, 1, 1, 1, config.nTileRow*config.nTileCol);

for t=1:opts.batchSize:numel(subset)
    fprintf('%s: epoch %02d: batch %3d/%3d: ', mode, epoch, ...
        fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
    batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
    batchTime = tic ;
    numDone = 0 ;
    res = [] ;
    res_syn = [];
%     stats = [] ;
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
                cell_idx = (ceil(t / opts.batchSize) - 1) * numlabs + labindex;
                syn_mat = gpuArray(syn_mats(:,:,:,:,cell_idx));
                syn_mat = langevin_dynamics_fast(config, net, syn_mat);
                syn_mats(:,:,:,:,cell_idx) = syn_mat;
                
                res = vl_simplenn(net, im, dydz, res, ...
                    'accumulate', s ~= 1, ...
                    'disableDropout', ~training, ...
                    'conserveMemory', opts.conserveMemory, ...
                    'backPropDepth', opts.backPropDepth, ...
                    'sync', opts.sync, ...
                    'cudnn', opts.cudnn);
                
                res_syn = vl_simplenn(net, gpuArray(syn_mat), dydz_syn, res_syn, ...
                    'accumulate', s ~= 1, ...
                    'disableDropout', ~training, ...
                    'conserveMemory', opts.conserveMemory, ...
                    'backPropDepth', opts.backPropDepth, ...
                    'sync', opts.sync, ...
                    'cudnn', opts.cudnn);  
                
            case {'cd_l2l', 'cd_e2e'}
                % compute one step contrast images
                % I1 = I + e^2/2 * (df/dI f(I;w) - I) + e Z; Z ~ N(0, 1)
                
                syn_mat = im;
                for t = 1:config.T
                    res = vl_simplenn(net, syn_mat, dydz, res, ...
                    'accumulate', s ~= 1, ...
                    'disableDropout', ~training, ...
                    'conserveMemory', opts.conserveMemory, ...
                    'backPropDepth', opts.backPropDepth, ...
                    'sync', opts.sync, ...
                    'cudnn', opts.cudnn) ;
                
                    syn_mat = syn_mat + ...
                        config.Delta^2/2 * (res(1).dzdx - syn_mat / config.refsig /config.refsig) + ...
                        config.Delta * gpuArray(randn(size(syn_mat), 'single'));
                end
                
%                 res = vl_simplenn(net, im, dydz, res, ...
%                     'accumulate', s ~= 1, ...
%                     'disableDropout', ~training, ...
%                     'conserveMemory', opts.conserveMemory, ...
%                     'backPropDepth', opts.backPropDepth, ...
%                     'sync', opts.sync, ...
%                     'cudnn', opts.cudnn) ;
%                 
%                 syn_mat = im + ...
%                     config.Delta^2/2 * (res(1).dzdx - im / config.refsig /config.refsig) + ...
%                     config.Delta * gpuArray(randn(size(im), 'single'));
                
                res_syn = vl_simplenn(net, syn_mat, dydz, res_syn, ...
                    'accumulate', s ~= 1, ...
                    'disableDropout', ~training, ...
                    'conserveMemory', opts.conserveMemory, ...
                    'backPropDepth', opts.backPropDepth, ...
                    'sync', opts.sync, ...
                    'cudnn', opts.cudnn) ;
        end
        numDone = numDone + numel(batch) ;
    end
    
    % gather and accumulate gradients across labs
    if training
        if numGpus <= 1
            [net, ~] = accumulate_gradients(opts, learningRate, batchSize, net, res, res_syn, config);
            if strcmp(config.model_type, 'sparse') == 1 ...
                    && strcmp(config.learn_scheme(1:3), 'hae')
                % sparse constraints
                res = vl_simplenn(net, im, [], [], 'conserveMemory', false, 'cudnn', 1);
                net = accumulate_bias(net, res, config);
                clear res;
            end
        else
            if isempty(mmap)
                mmap = map_gradients(opts.memoryMapFile, net, res, res_syn, numGpus) ;
            end
            
            write_gradients(mmap, net, res, res_syn) ;
            labBarrier() ;
            [net, ~] = accumulate_gradients(opts, learningRate, batchSize, net, res, res_syn, config, mmap) ;
            
            if strcmp(config.model_type, 'sparse') == 1 ...
                    && strcmp(config.learn_scheme(1:3), 'hae')
                % sparse constraints
                res = vl_simplenn(net, im, [], [], 'conserveMemory', false, 'cudnn', 1);
                if isempty(mmap_bias)
                    mmap_bias = map_bias([opts.memoryMapFile(1:end-4), '_bias.bin'], net, res, numGpus) ;
                end
                
                write_bias(mmap_bias, net, res) ;
                labBarrier();
                net = accumulate_bias(net, res, config, mmap_bias);
                clear res;
            end
        end
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
end


% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(opts, lr, batchSize, net, res, res_syn, config, mmap)
% -------------------------------------------------------------------------
layer_sets = config.layer_sets;
% if nargin < 8
%     layer_sets = numel(net.layers):-1:numel(net.layers)-2;
% end
num_syn = config.nTileRow * config.nTileCol;

for l = layer_sets
    for j=1:numel(res(l).dzdw)
        thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
        thisLR = lr * net.layers{l}.learningRate(j) ;
        
        % accumualte from multiple labs (GPUs) if needed
        if nargin >= 8
            tag = sprintf('l%d_%d',l,j) ;
            tmp = zeros(size(mmap.Data(labindex).(tag)), 'single') ;
            for g = setdiff(1:numel(mmap.Data), labindex)
                tmp = tmp + mmap.Data(g).(tag) ;
            end
            res(l).dzdw{j} = res(l).dzdw{j} + tmp ;
            
            if ~isempty(res_syn)
                tag = sprintf('syn_l%d_%d',l,j) ;
                tmp = zeros(size(mmap.Data(labindex).(tag)), 'single') ;
                for g = setdiff(1:numel(mmap.Data), labindex)
                    tmp = tmp + mmap.Data(g).(tag) ;
                end
                res_syn(l).dzdw{j} = res_syn(l).dzdw{j} + tmp ;
            end
        end
        
        if isfield(net.layers{l}, 'weights')
            
            switch config.learn_scheme
                case {'mle_l2l', 'mle_e2e'}
                    gradient_dzdw = ((1 / batchSize) * res(l).dzdw{j} -  ...
                        (1 / num_syn) * res_syn(l).dzdw{j}) / net.numFilters(l);
                    if max(abs(gradient_dzdw(:))) > 20 %10
                        gradient_dzdw = gradient_dzdw / max(abs(gradient_dzdw(:))) * 20;
                    end

                case {'cd_l2l', 'cd_e2e'}
                    gradient_dzdw = ((1 / batchSize) * res(l).dzdw{j} -  ...
                        (1 / batchSize) * res_syn(l).dzdw{j}) / net.numFilters(l);
                   
                    if max(gradient_dzdw(:)) > 1 %102
                        gradient_dzdw = gradient_dzdw / max(gradient_dzdw(:)) * 1;
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
                
                if strcmp(config.model_type, 'sparse') == 1
                    sparse_ratio = sum( res(res_l).x(:) == 0) / length(res(res_l).x(:));
                    fprintf('The sparse ratio is %.2f\n', sparse_ratio);
                end
            end
            %       net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR * gradient_dzdw;
        end
    end
end
end

function mmap = map_gradients(fname, net, res, res_syn, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.layers)
    for j=1:numel(res(i).dzdw)
        format(end+1,1:3) = {'single', size(res(i).dzdw{j}), sprintf('l%d_%d',i,j)} ;
    end
end

if ~isempty(res_syn)
    fprintf('writting res_syn!!\n');
    for i=1:numel(net.layers)
        for j=1:numel(res_syn(i).dzdw)
            format(end+1,1:3) = {'single', size(res_syn(i).dzdw{j}), sprintf('syn_l%d_%d',i,j)} ;
        end
    end
end

format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname, 'file') && (labindex == 1)
    f = fopen(fname,'wb') ;
    for g=1:numGpus
        for i=1:size(format,1)
            fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
        end
    end
    fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;
end

function write_gradients(mmap, net, res, res_syn)
% -------------------------------------------------------------------------
for i=1:numel(net.layers)
    for j=1:numel(res(i).dzdw)
        mmap.Data(labindex).(sprintf('l%d_%d',i,j)) = gather(res(i).dzdw{j}) ;
    end
end

if ~isempty(res_syn)
    for j=1:numel(res_syn(i).dzdw)
        mmap.Data(labindex).(sprintf('syn_l%d_%d',i,j)) = gather(res_syn(i).dzdw{j}) ;
    end
end
end

function net = accumulate_bias(net, res, config, mmap)
layer_sets = config.layer_sets;
% if nargin < 8
%     layer_sets = numel(net.layers):-1:numel(net.layers)-2;
% endwrite_bias

for l = layer_sets
    % accumualte from multiple labs (GPUs) if needed
    res_l = [];
    if nargin >= 4
        tag = sprintf('l%d',l) ;
        for g = setdiff(1:numel(mmap.Data), labindex)
            res_l = cat(4, res_l, mmap.Data(g).(tag));
        end
    end
    
    if isfield(net.layers{l}, 'weights') ...
            && strcmp(net.layers{l}.name(1:2), 'fc') == 0
        
        res_l = cat(4, res_l, res(l+1).x);
        sz = size(res_l);
        res_l = reshape(reshape(res_l, [], sz(4))', [], sz(3));
        bias = single(prctile(res_l, 100-config.sparse_level(l)));
        net.layers{l}.weights{2} = net.layers{l}.weights{2} - bias;
    end
end
end


function mmap = map_bias(fname, net, res, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for l=1:numel(net.layers)
    if isfield(net.layers{l}, 'weights')
        format(end+1,1:3) = {'single', size(res(l+1).x), sprintf('l%d',l)} ;
    end
end

if ~exist(fname, 'file') && (labindex == 1)
    f = fopen(fname,'wb') ;
    for g=1:numGpus
        for i=1:size(format,1)
            fwrite(f, zeros(format{i,2}, format{i,1}), format{i,1}) ;
        end
    end
    fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;
end


function write_bias(mmap, net, res)
% -------------------------------------------------------------------------
for l=1:numel(net.layers)
    if isfield(net.layers{l}, 'weights')
        mmap.Data(labindex).(sprintf('l%d',l)) = gather(res(l+1).x) ;
    end
end
end
 