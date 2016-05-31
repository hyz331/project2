function [] = merge_images()

img_dir = '/home/yanglu/Dropbox/Paper/CNNFRAMEfinal2/figures/result_layer_16_4_filters/painting/';
space = 5;
syn_list = dir([img_dir, '0*.png']);
num_syn = numel(syn_list);
canvas = zeros(224, 224*(num_syn+1) + space, 3, 'uint8');

train_img = imread([img_dir, 'train001.jpg']);
train_img = imresize(train_img, [224,224]);
canvas(:,1:224,:) = train_img;

for j = 1:num_syn
    img = imread([img_dir, syn_list(j).name]);
    img = imresize(img, [224,224]);
    canvas(:,j*224+space + 1:j*224+space + 224,:) = img;
end

imwrite(canvas, [img_dir, 'canvas.png']);