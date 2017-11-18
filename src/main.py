from src.Text_Detection.Detection import Detection
from src.Character_Segmentation.Segmentation import Segmentation
from src.utils import pre_process as pp
import argparse
import cv2

file = open("Transcript.txt", "w")
# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ip", "--imagePath", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

# Pre Process The Image ( G-Scale, Resize etc )
processed_image = pp.process_new_img(image=image, widht=750)

# Text Detection Operation #
dt = Detection(image=processed_image, win_w=24, win_h=12)
sg = Segmentation()
# Iterate over the detected patches and implement segmentation operation
for patch in dt.draw_rectangles():
    coord_list = sg.segment_patch(patch=patch)
    for char in Segmentation.separate_chars(patch=patch, coordlist=coord_list):
        # Run classifier for every char.
        result = None # It will have a value after classifier works
        file.write(result)

file.close()
