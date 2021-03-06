function net = train_model_cd(config, net, imdb, getBatch, layer)

rate_list = ones(1,60, 'single');%logspace(-2, -4, 120)*100; %ones(1,60, 'single');
learningRate_array = repmat(rate_list , floor(config.nIteration / length(rate_list)),1); %logspace(-2, -4, 60) ones(1,60, 'single')
learningRate_array = reshape(learningRate_array, 1, []);

opts.batchSize = config.BatchSize ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = config.gpus; % which GPU devices to use (none, one, or more)
opts.learningRate = 0.001 ;
opts.continue = false ;
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.prefetch = false;
opts.numFetchThreads = 8;
opts.cudnn = true ;
opts.weightDecay = 0.0001 ; %0.0001
opts.momentum = 0.9 ;
opts.errorFunction = 'multiclass' ;
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;
opts.memoryMapFile = fullfile(config.working_folder, 'matconvnet.bin') ;
opts.learningRate = reshape(learningRate_array, 1, []);
% opts = vl_argparse(opts, varargin) ;


% if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

opts.batchSize = min(opts.batchSize, numel(opts.train));
opts.numEpochs = config.nIteration;

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

net.filterSelected = 1:prod(config.dydz_sz);
net.selectedLambdas = ones(1, prod(config.dydz_sz), 'single') * 1;

interval = ceil(opts.numEpochs / 100);
SSD = zeros(opts.numEpochs, 1);


% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
  if isempty(gcp('nocreate')),
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
elseif numGpus == 1
  gpuDevice(opts.gpus)
end
% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------
h = figure;
for epoch=1:opts.numEpochs
    fprintf('Layer %d / %d, iteration %d / %d\n', layer, config.layer_to_learn, epoch, opts.numEpochs);
    
    learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) * config.Gamma;
    
    % train one epoch and validate
    train = opts.train(randperm(numel(opts.train))) ; % shuffle
    
    if numGpus <= 1
        net = process_epoch_generative(opts, getBatch, epoch, train, learningRate, imdb, net, [], config);
        loss = computerLoss(opts, config, imdb, getBatch, train, net);
    else
        spmd(numGpus)
            net_ = process_epoch_generative(opts, getBatch, epoch, train, learningRate, imdb, net, [], config);
            loss_ = gplus(computerLoss(opts, config, imdb, getBatch, train, net));
        end
        net = net_{1};
        loss = loss_{1};
        clear net_;
        clear loss_;
    end
    
    SSD(epoch) = loss;
    
    
    disp(['Loss: ', num2str(SSD(epoch))]);
    if mod(epoch - 1, interval) == 0 || epoch == opts.numEpochs
        % save samples
        recons = train(randperm(numel(train)));
        recons = recons(1: min(config.nTileRow * config.nTileCol, numel(recons)));
        %     syn_mat = compute_syn(config, net, imdb, getBatch, recons);
        syn_mat = compute_recons(opts, config, net, imdb, getBatch, recons);
        
        
        
        draw_figures_hae(config, syn_mat, epoch, net.normalization.averageImage, SSD, layer);
        
        
        if mod(epoch, 100) == 0
            model_file = [config.working_folder, num2str(layer, 'layer_%02d'), '_iter_',...
                num2str(epoch) ,'_model.mat'];
            save(model_file, 'net')
            
            saveas(h, [config.working_folder, num2str(layer, 'layer_%02d_'), '_iter_',...
                num2str(epoch) ,'_error.fig']);
            saveas(h, [config.working_folder, num2str(layer, 'layer_%02d_'), '_iter_',...
                num2str(epoch) ,'_error.png'])
        end
    end
end
end

function recon_imgs = compute_recons(opts, config, net_cpu, imdb, getBatch, subset)
net = vl_simplenn_move(net_cpu, 'gpu');
recon_imgs = zeros([size(net.normalization.averageImage), numel(subset)], 'single');
numGpus = numel(opts.gpus) ;
res = [];
for t=1:opts.batchSize:numel(subset)
    batchStart = t;
    batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : batchEnd) ;
    im = getBatch(imdb, batch);
    
    if numGpus >= 1
        im = gpuArray(im) ;
    end
    
    % training images
    numImages = size(im, 4);
    
    dydz = gpuArray(zeros(config.dydz_sz, 'single'));
    dydz(net.filterSelected) = net.selectedLambdas;
    dydz = repmat(dydz, 1, 1, 1, numImages);
    
%     syn_mat = im;
%     for t = 1:config.T       
%         res = vl_simplenn(net, syn_mat, dydz, res, 'conserveMemory', 1, 'cudnn', 1);
%         
%         syn_mat = syn_mat + ...
%             config.Delta^2/2 * (res(1).dzdx - syn_mat / config.refsig /config.refsig) + ...
%             config.Delta * gpuArray(randn(size(syn_mat), 'single'));
%     end
%     
%     recon_imgs(:,:,:,batchStart:batchEnd) = gather(syn_mat);

    res = vl_simplenn(net, gpuArray(im), dydz, [], 'conserveMemory', 1, 'cudnn', 1);
    recon_imgs(:,:,:,batchStart:batchEnd) = gather(res(1).dzdx) * config.refsig^2;
end
clear net;
end


function recon_imgs = compute_syn(config, net_cpu, imdb, getBatch, subset)
net = vl_simplenn_move(net_cpu, 'gpu');
recon_imgs = zeros([size(net.normalization.averageImage),...
    config.nTileRow * config.nTileCol], 'single');

batchStart = 1;
batchEnd =numel(subset);
batch = subset(batchStart : batchEnd) ;
im = getBatch(imdb, batch);

syn_mat = repmat(im, [1,1,1,floor(config.nTileRow * config.nTileCol / size(im, 4))]);
syn_mat = langevin_dynamics_fast(config, net, syn_mat);
recon_imgs(:,:,:,1:size(syn_mat, 4)) = syn_mat;

clear net;
end



 