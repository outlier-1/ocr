import random
import scipy.io as sio
import os
import numpy as np
import glob
import cv2
from PIL import Image

# def createdata(x, y):
#     a=0
#     for filename in glob.glob("rectangles/" + '*.png'):
#         list_helper = 0
#         tmp_image = cv2.imread(filename)
#         for resized in pyramid(tmp_image):
#             # loop over the sliding window for each layer of the pyramid
#             pyramid_counter=0
#             for (x, y, window) in sliding_windows(resized, stepSize=1, windowSize=(7, 30), startX=0, startY=0):
#                 # if the window does not meet our desired window size, ignore it
#                 if window.shape[0] != 7 or window.shape[1] != :
#                     continue
#                 clone = resized.copy()
#                 crop_img = clone[y:y + winH, x:x + winW]
#                 cv2.imwrite(dataset_path + name_generator() + '.png', crop_img)
#                 list_helper += 1
#                 # Draw a Rectangle
#                 cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
#                 # if y + winH > 994:
#                 #     break
#                 cv2.imshow("Window", clone)
#                 cv2.waitKey(1) # unnecessary for real-time work
#             pyramid_counter += 1
#             print(list_helper)
#             break
#
#
# def create_dataset_files(totalimg):
#     for filename in glob.glob(image_path + '*8S.jpg'):
#         list1 = randomDeterminer(0)
#         list2 = randomDeterminer(1)
#         list3 = randomDeterminer(2)
#         list4 = randomDeterminer(3)
#         listOfChoices = [list1, list2, list3, list4]
#         list_helper = 0
#         tmp_image = cv2.imread(filename)
#         for resized in pyramid(tmp_image):
#             # loop over the sliding window for each layer of the pyramid
#             pyramid_counter=0
#             for (x, y, window) in sliding_windows(resized, stepSize=8, windowSize=(winW, winH), startX=0, startY=0):
#                 # if the window does not meet our desired window size, ignore it
#                 if window.shape[0] != winH or window.shape[1] != winW:
#                     continue
#                 clone = resized.copy()
#                 # if list_helper in listOfChoices[pyramid_counter]:
#                 #     crop_img = clone[y:y + winH, x:x + winW]
#                 #     cv2.imwrite(dataset_path + name_generator() + '.png', crop_img)
#                 list_helper += 1
#                 # Draw a Rectangle
#                 cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
#                 cv2.imshow("Window", clone)
#                 #cv2.waitKey(10) unnecessary for real-time work
#             pyramid_counter += 1
#             list_helper = 0
#         totalimg-=1
#         print("{} Image Left..".format(str(totalimg)))

dataset_matrix = np.zeros(shape=(5000, 449))
# CREATE A GIANT DATASET MATRIX FROM FILES ( M x 6144 )
def create_dataset_matrix(T, featureSize):
    startIndex = 0
    for filename in glob.glob('../Character_Segmentation/ds/' + '*.png'):
        img = cv2.imread(filename)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tmp_array = np.array(image).reshape(1, featureSize)
        dataset_matrix[startIndex, 1:] = tmp_array
        startIndex += 1
        if startIndex % 500 == 0:
            print("%{}. Completed..".format(startIndex/50))
    print("Dataset Matrix Succesfully Created.. Size Of: {}x{}".format(T, featureSize))
    return dataset_matrix


def create_matfile():
    ds = create_dataset_matrix(5000, 448)
    convert_type = np.array(ds, dtype='uint8')
    # for i in range(50):
    #     ds = shuffle(ds)
    #     print("Shuffled.. {}.Th Time".format(i+1))
    sio.savemat('Train.mat', mdict={'dataset': convert_type})

# GOTTA FIX
def determine_labels():
    mat = sio.loadmat('Train.mat')
    ds = mat.get('dataset')
    for x in range(0, 5001):
        if x % 250 == 0 and x != 0:
            print("Saving Changes...")
            print("{}% Completed..".format((x/5000)*100))
            sio.savemat('Train.mat', mdict={'dataset': ds})
            keep = input("You've Labeled 250 Records.. \nIf you wanna keep labeling, press 'y', otherwise 'n'\nYour Decision: ")
            if keep == 'y':
                print("Keep going..")
            else:
                print("Last record's index is: " + str(x - 1) + "\nPlease Save That..")
                exit(0)
        tmp = ds[x, 1:].reshape(32, 14)
        img = Image.fromarray(tmp)
        img.show()
        ans = input("Your Answer: \n")
        while True:
            if ans.isdigit():
                ds[x, 0] = int(ans)
                os.system('xdotool search "ImageMagick:" windowkill %@')
                break
            else:
                ans = input("Enter an integer.")
                # i=0, take i'th row and reshape as x. imshow x
                # wait for input, take input and assign it to i'th row of Label Vector.
    print("saving changes..")
    sio.savemat('Train.mat', mdict={'dataset': ds})
    print("done.")


# def random_determiner(pyramidIndex):
#     choices = []
#     start = 1
#     if pyramidIndex == 0:
#         while True:
#             first_pyramid = list(range(0, 4698))  # 447
#             tmp = random.choice(first_pyramid)
#             if tmp in choices:
#                 continue
#             else:
#                 choices.append(tmp)
#             if len(choices) == 447:
#                 return choices
#
#     elif pyramidIndex == 1:
#         while True:
#             second_pyramid = list(range(0, 2925))  # 278
#             tmp = random.choice(second_pyramid)
#             if tmp in choices:
#                 continue
#             else:
#                 choices.append(tmp)
#             if len(choices) == 278:
#                 return choices
#
#     elif pyramidIndex == 2:
#         while True:
#             third_pyramid = list(range(0, 1785))  # 170
#             tmp = random.choice(third_pyramid)
#             if tmp in choices:
#                 continue
#             else:
#                 choices.append(tmp)
#             if len(choices) == 170:
#                 return choices
#
#     elif pyramidIndex == 3:
#         while True:
#             forth_pyramid = list(range(0, 1107))  # 105
#             tmp = random.choice(forth_pyramid)
#             if tmp in choices:
#                 continue
#             else:
#                 choices.append(tmp)
#             if len(choices) == 105:
#                 return choices
#
#     else:
#         print("Input a valid parameter.")
#
# def resizeImage():
#     b=0
#     for filename in glob.glob('path' + '*.png'):
#         a = cv2.imread(filename)
#         b+=1
#         if a.shape[0] != 32:
#             res = imutils.resize(a, width=64, height=32)
#             cv2.imwrite(filename,res)

determine_labels()
