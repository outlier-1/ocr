import cv2
import numpy as np
import imutils
import string

def process_new_img(imagePath:string, destPath:string, name:string, widht=750, gscaled=False):
    image = cv2.imread(imagePath)  # Read image
    #  print(type(image))
    if image is None:
        print("Could not found the image. Check the imagePath again. ")
        return None
    else:
        if gscaled:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert To Gray-Scale
        processed = imutils.resize(image, width=widht, inter=cv2.INTER_AREA)  # Resize
        cv2.imwrite(destPath + name, processed)  # Write Image
        return processed


def rgb2gray(img):
    row, columns, channels = img.shape
    if channels == 3:  # Convert this image to grayscale from RGB
        redChannel = img[:, :, 0]
        greenChannel = img[:, :, 1]
        blueChannel = img[:, :, 2]
        grayImage = (0.2989 * redChannel) + (0.5870 * greenChannel) + (0.1140 * blueChannel)  # ans is A*B matrix
        grayImage = np.array(grayImage, dtype='uint8')
        return grayImage
    else:
        print('Img is already gray-scaled.')


def zero_mean(*args, **kwargs):
    retlist = []
    for arg in args:
        arr = np.asarray(arg, dtype=np.float32)
        accRed = 0
        accGreen = 0
        accBlue = 0

        for i in range(0, arr.shape[0]):
            tmp = arr[i, :].reshape(32, 64, 3)
            red = tmp[:, :, 0]
            green = tmp[:, :, 1]
            blue = tmp[:, :, 2]
            accRed += np.sum(red)
            accGreen += np.sum(green)
            accBlue += np.sum(blue)

        meanRed = accRed / (32 * 64 * arr.shape[0])
        meanGreen = accGreen / (32 * 64 * arr.shape[0])
        meanBlue = accBlue / (32 * 64 * arr.shape[0])

        for i in range(0, arr.shape[0]):
            tmp = arr[i, :].reshape(32, 64, 3)
            tmp[:, :, 0] = tmp[:, :, 0] - meanRed
            tmp[:, :, 1] = tmp[:, :, 1] - meanGreen
            tmp[:, :, 2] = tmp[:, :, 2] - meanBlue
            a = tmp.reshape(1, 6144)
            arr[i, :] = a

        retlist.append(arr)
    return retlist

process_new_img(imagePath='IMG_0620.JPG', destPath='d', name='okay.jpg')
