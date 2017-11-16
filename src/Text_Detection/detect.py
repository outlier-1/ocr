# import the necessary packages
import cv2
import math
from src.Text_Detection.utils import utils as u

text_patch_list = []


def draw_windows(real_img, blueprint):
    img = blueprint

    pointer_X = -1
    pointer_Y = -1

    coordlist = []

    while pointer_X < (img.shape[0] - 10):
        pointer_X += 1
        if 255 in img[pointer_X, :, :]:
            x1 = pointer_X
            while True:
                pointer_X += 1
                if 255 not in img[pointer_X, :, :] or pointer_X - x1 > 40:
                    x2 = pointer_X
                    img[pointer_X:pointer_X + 5, :, :] = 0
                    break

            while pointer_Y < (img.shape[1] - 1):
                pointer_Y += 1
                if 255 in img[x1:x2, pointer_Y, :]:
                    y1 = pointer_Y
                    while True:
                        pointer_Y += 1
                        if 255 not in img[x1:x2, pointer_Y:pointer_Y + 10, :]:
                            y2 = pointer_Y
                            if int(math.fabs(x1 - x2)) / int(math.fabs(y1 - y2)) > 5:
                                break
                            else:
                                coordlist.append([x1, x2, y1, y2])
                            break
            pointer_Y = -1

    for i in range(0, len(coordlist)):
        # noinspection PyUnresolvedReferences
        cv2.rectangle(real_img, ((coordlist[i][2]) + 5, coordlist[i][0]),
                      ((coordlist[i][3]) + 5, (coordlist[i][1])), (0, 255, 0), 2)
        patch = real_img[coordlist[i][0]:coordlist[i][1], coordlist[i][2] + 5:coordlist[i][3] + 5]
        if (patch.shape[0] == 0) or (patch.shape[1] == 0):
            pass
        else:
            # noinspection PyUnresolvedReferences
            text_patch_list.append(patch)
            print("Added..{}".format(i))
            cv2.imwrite("rectangles/" + u.name_generator(size=4) + ".png", patch)
    return real_img


