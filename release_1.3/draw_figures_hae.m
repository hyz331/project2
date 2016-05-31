function [] = draw_figures_hae(config, recon_imgs, iter, mean_img, SSD, layer)

config.nTileRow = 6;
config.nTileCol = 6;

num_imgs = size(recon_imgs, 4);
config.nTileRow = min(ceil(sqrt(num_imgs)),config.nTileRow);
config.nTileCol = config.nTileRow;

if num_imgs > config.nTileRow * config.nTileCol
   subset = randperm(num_imgs);
   subset = subset(1:config.nTileRow * config.nTileCol);
   recon_imgs = recon_imgs(:,:,:,subset);
end

[I_recon, recon_mat_norm] = convert_syns_mat(config, mean_img, recon_imgs);
for i = 1:size(recon_mat_norm, 4)
    imwrite(recon_mat_norm(:,:,:,i), [config.figure_folder, num2str(layer, 'layer_%02d_'), num2str(i, '%03d.png')]);
end
im = im2uint8(I_recon);
[imind,cm] = rgb2ind(im,256);
% save samples
if iter == 1 && layer == 1
    imwrite(imind, cm, [config.Synfolder, 'animation', '.gif'], 'DelayTime', 0.10, 'Loopcount', inf);
else
    imwrite(imind, cm, [config.Synfolder, 'animation', '.gif'], 'WriteMode', 'append', 'DelayTime', 0.10);
end

imwrite(I_recon, [config.Synfolder, num2str(layer, 'layer_%02d_'), num2str(iter, 'dense_original_%04d'), '.png']);
plot(1:iter, SSD(1:iter), 'r', 'LineWidth', 3);
axis([min(iter, 1), iter+1, min(SSD(min(iter, 1):iter)) - 0.2*abs(min(SSD(min(iter, 1):iter))),  max(SSD(min(iter, 1):end)) * 1.2]);
title('Loss')
drawnow;
end