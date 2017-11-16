function path = binarizeImage(filename,out)
tmp = imread(filename);
gscaled = rgb2gray(tmp);
img = imbinarize(gscaled,'adaptive','ForegroundPolarity','dark','Sensitivity',0.4);
imwrite(img,out);
path = out;
