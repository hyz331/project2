function [loss, recon_imgs] = computerLoss(opts, config, imdb, getBatch, subset, net_cpu)

numGpus = numel(opts.gpus) ;
net = vl_simplenn_move(net_cpu, 'gpu') ;

loss = 0;

for t=1:opts.batchSize:numel(subset)
    for s=1:opts.numSubBatches
        batchStart = t + (labindex-1) + (s-1) * numlabs ;
        batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
        im = getBatch(imdb, batch);
        
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

        res = vl_simplenn(net, gpuArray(im), dydz, [], 'conserveMemory', 1, 'cudnn', 1);
        recon_imgs = res(1).dzdx * config.refsig^2;
        loss = loss + gather(mean(reshape((recon_imgs - im).^2, [], 1)));
    end
end
loss = loss / numel(subset);
clear net;
end