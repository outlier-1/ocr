# import the necessary packages
import imutils
import cv2
import numpy as np
import math
from src.Text_Detection.Detection_Classifier import DetectionClassifier
from src.utils import pre_process as pp


class Detection:
    def __init__(self, image, win_w, win_h):
        self.image = image
        self.win_w = win_w
        self.win_h = win_h
        self.blueprint = self.set_blueprint()
        self.coordlist = []

    def sliding_windows(self, step_size, window_size):
        # slide a window across the image
        for y in range(0, self.image.shape[0], step_size):
            for x in range(0, self.image.shape[1], step_size):
                # yield the current window
                yield (x, y, self.image[y:y + window_size[1], x:x + window_size[0]])

    # noinspection PyUnresolvedReferences
    def set_blueprint(self, expansion_operation=True, debug_mode=False):
        tmp_blueprint = np.zeros(shape=(self.image.shape[0], self.image.shape[1], self.image.shape[2]))
        cl = DetectionClassifier()
        for (x, y, window) in self.sliding_windows(step_size=16, window_size=(self.win_w, self.win_h)):
            if window.shape[0] != self.win_h or window.shape[1] != self.win_w:
                continue

            clone = self.image.copy()

            # Take the window, resize the appropraite size and give it to the classifier
            patch = clone[y:y + self.win_h, x:x + self.win_w]
            resized_patch = imutils.resize(patch, width=64, height=32)
            input_for_classifier = pp.zero_mean(resized_patch.reshape(1, 6144))

            prediction = cl.make_predict(input_for_classifier[0])

            # If classifier detects text, mark these pixels as white, otherwise black
            if prediction == 1:
                tmp_blueprint[y:y + self.win_h, x:x + self.win_w] = 255
                print("hey")
            else:
                tmp_blueprint[y:y + self.win_w, x:x + self.win_w] = 0

            # Draw a Rectangle
            if debug_mode:
                cv2.rectangle(clone, (x, y), (x + self.win_w, y + self.win_h), (0, 255, 0), 2)
                cv2.imshow("window", clone)
                cv2.waitKey(10)
        if expansion_operation:
            tmp_blueprint = self.expansion(tmp_blueprint, pixels=3)
        return tmp_blueprint

    @staticmethod
    def expansion(blueprint, pixels=3):
        for y in range(blueprint.shape[0]):
            for x in range(blueprint.shape[1]):
                if blueprint[y, x, 1] == 255 and x < blueprint.shape[1] - pixels:
                    if blueprint[y, x + pixels, 1] == 255:
                        for i in range(1, pixels):
                            blueprint[y, x + i, :] = 255
        return blueprint

    def draw_rectangles(self, debug_mode=False):
        self.coordlist = []
        pointer_X = -1
        pointer_Y = -1

        while pointer_X < (self.blueprint.shape[0] - 10):
            pointer_X += 1
            if 255 in self.blueprint[pointer_X, :, :]:
                x1 = pointer_X
                while True:
                    pointer_X += 1
                    if 255 not in self.blueprint[pointer_X, :, :] or pointer_X - x1 > 40:
                        x2 = pointer_X
                        self.blueprint[pointer_X:pointer_X + 5, :, :] = 0
                        break

                while pointer_Y < (self.blueprint.shape[1] - 1):
                    pointer_Y += 1
                    if 255 in self.blueprint[x1:x2, pointer_Y, :]:
                        y1 = pointer_Y
                        while True:
                            pointer_Y += 1
                            if 255 not in self.blueprint[x1:x2, pointer_Y:pointer_Y + 10, :]:
                                y2 = pointer_Y
                                if int(math.fabs(x1 - x2)) / int(math.fabs(y1 - y2)) > 5:
                                    break
                                else:
                                    self.coordlist.append([x1, x2, y1, y2])
                                break
                pointer_Y = -1

        for i in range(0, len(self.coordlist)):
            if debug_mode:
                # noinspection PyUnresolvedReferences
                cv2.rectangle(self.image, ((self.coordlist[i][2]) + 5, self.coordlist[i][0]),
                              ((self.coordlist[i][3]) + 5, (self.coordlist[i][1])), (0, 255, 0), 2)

            patch = self.image[self.coordlist[i][0]:self.coordlist[i][1],
                    self.coordlist[i][2] + 5:self.coordlist[i][3] + 5]

            if (patch.shape[0] == 0) or (patch.shape[1] == 0):
                pass
            else:
                yield patch
