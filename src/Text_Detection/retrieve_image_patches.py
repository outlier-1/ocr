# import the necessary packages
import imutils
import cv2
import numpy as np
from src.Text_Detection.detect import detect_text
import classifier as cl
import utils.pre_process as pp

(winW, winH) = (32, 16)
image_path = 'PreProcessed_Images/'

startIndex = 0

def pyramid(image, scale=1, minSize=(350, 350), isActive=False):
    if isActive:
        yield image
        while True:
            w = int(image.shape[1] / scale)
            image = imutils.resize(image, width=w)
            if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
                break
            yield image
    else:
        yield image

def sliding_windows(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

image = cv2.imread('PreProcessed_Images/aokay.jpg')
newimg = np.zeros(shape=(1060,750,3))
def start():
    for resized in pyramid(image, scale=1):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_windows(resized, stepSize=16, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            clone = resized.copy()
            a =(0, 255, 0)
            img = clone[y:y+winH, x:x+winW]
            resize = imutils.resize(img, width=64, height=32)
            reshape = resize.reshape(1,6144)
            pa = pp.zero_mean(reshape)
            pred = cl.makePredict(pa[0])
            if pred==1:
                a = (0,255,0)
                newimg[y:y + winH, x:x + winW]=255
                # if 255 in newimg[y:y + winH, x:x + winW]:
                #     continue
                # else:
                #
            elif pred==0:
                a = (0,0,255)
                newimg[y:y + winH, x:x + winW]=0

            # since we do not have a classifier, we'll just draw the window
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), a, 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
        break
start()
def edit(blue_print):
    for x in range(1060):
        for y in range(750):
            if blue_print[x, y, 1] == 255 and y<1055:
                if blue_print[x , y+5 , 1] == 255:
                    for i in range(1, 5):
                        blue_print[x, y + i, :] = 255
    for y in range(750):
        for x in range(1060):
            if blue_print[x, y, 1] == 255 and x<1055:
                if blue_print[x + 5, y, 1] == 255:
                    for i in range(1, 5):
                        blue_print[x + i, y, :] = 255

    cv2.imwrite("utils/done.jpg", blue_print)
    return blue_print

newimg = cv2.imread('utils/done.jpg')

blue_print = edit(newimg)

realim = detect_text(blueprint=blue_print, realim=image)
cv2.imshow('window', realim)
cv2.waitKey(1000)
