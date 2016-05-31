function [] = split_canvas(img, nRow, nCol, dst_dir)

if nargin < 1
    img = imread(['/home/yanglu/Projects/autoEncoder/Image/korean_face/total.jpg']);
end

if nargin < 2
    nRow = 3;
end

if nargin < 3
    nCol = 6;
end

if nargin < 4
   dst_dir = '/home/yanglu/Projects/autoEncoder/Image/korean_face/'; 
end

sz = size(img);
h = sz(1);
w = sz(2);
sx = floor(w / nCol);
sy = floor(h / nRow);


k = 1;
for i = 1:nRow
    for j = 1:nCol
        splited = img( 1+(i-1)*sy:sy+(i-1)*sy, 1+(j-1)*sx:sx+(j-1)*sx, :);
        imwrite(splited, [dst_dir, num2str(k, '%03d.jpg')]);
        k = k+1;
    end
end