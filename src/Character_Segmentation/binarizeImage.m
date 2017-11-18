function path = binarizeImage(imagePath,out)
image = imread(imagePath);
gscaled = rgb2gray(image);
img = imbinarize(gscaled,'adaptive','ForegroundPolarity','dark','Sensitivity',0.4);
imwrite(img,out);
path = out;
