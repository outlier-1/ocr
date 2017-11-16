# import the necessary packages
import imutils
import cv2
import numpy as np
from src.Text_Detection.detect import detect_text
import classifier as cl
import utils.pre_process as pp

(winW, winH) = (64, 32)


def sliding_windows(image, step_size, window_size):
    # slide a window across the image
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            # yield the current window
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


# noinspection PyUnresolvedReferences
def get_blueprint(filename: string):
    image = cv2.imread(filename)
    blueprint = np.zeros(shape=(image.shape[0], image.shape[1], image.shape[2]))
    for (x, y, window) in sliding_windows(image, step_size=36, window_size=(winW, winH)):

        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        clone = image.copy()

        # Take the window, resize the appropraite size and give it to the classifier
        patch = clone[y:y + winH, x:x + winW]
        resized_patch = imutils.resize(patch, width=64, height=32)
        input_for_classifier = pp.zero_mean(resized_patch.reshape(1, 6144))
        prediction = cl.makePredict(input_for_classifier[0])

        # If classifier detects text, mark these pixels as white, otherwise black
        if prediction == 1:
            blueprint[y:y + winH, x:x + winW] = 255
        else:
            blueprint[y:y + winH, x:x + winW] = 0

        # Draw a Rectangle
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("window", clone)
        cv2.waitKey(10)
        return blueprint


def expanse(blue_print, pixels=3):
    for y in range(blue_print.shape[0]):
        for x in range(blue_print.shape[1]):
            if blue_print[y, x, 1] == 255 and x < blue_print.shape[1] - pixels:
                if blue_print[y, x + pixels, 1] == 255:
                    for i in range(1, pixels):
                        blue_print[y, x + i, :] = 255
    return blue_print
