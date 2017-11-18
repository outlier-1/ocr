import cv2
import matlab.engine


class Segmentation:
    def __init__(self):
        self.eng = matlab.engine.start_matlab()
        self.tmp_path = 'tmp/'
        self.coordinate_list = []

    def segment_patch(self, patch):
        self.coordinate_list = []
        cv2.imwrite(self.tmp_path + 'a.png', patch)
        pointer_X = 0
        binarized_rect = cv2.imread(self.eng.binarizeImage(self.tmp_path + "a.png", self.tmp_path + "b.png"))
        y_one, y_two = 0, patch.shape[0]

        while pointer_X < binarized_rect.shape[1]:
            # Look for first occur
            pointer_X += 1
            control = False
            # Look for character without space,
            if 0 in binarized_rect[:, pointer_X - 1]:
                x_one = pointer_X - 1
                x_two = x_one

                # Check for last 10 pixels, if line drew, don't draw new line, otherwise draw.
                for a in range(1, 10):
                    if [x_one - a, 0, x_one - a, y_two] in self.coordinate_list:
                        control = True
                if control is False:
                    self.coordinate_list.append([x_one, y_one, x_two, y_two])

                # Look for end of the character
                while True:
                    pointer_X += 1
                    # Draw a line to the end of the character.
                    if 0 not in binarized_rect[:, pointer_X]:
                        x_one = pointer_X + 1
                        x_two = x_one
                        self.coordinate_list.append([x_one, y_one, x_two, y_two])
                        break
        return self.coordinate_list

    @staticmethod
    def draw_lines(patch, coordinate_list):
        for item in coordinate_list:
            cv2.line(patch, (item[0], item[1]), (item[2], item[3]), (0, 255, 0), 1)

    @staticmethod
    def separate_chars(patch, coordlist):
        for item in range((len(coordlist) - 1)):
            character = patch[:, coordlist[item][0]:coordlist[item + 1][0]]
            yield character


x = cv2.imread('../Text_Detection/rectangles/A73.png')
a = Segmentation()
coord = a.segment_patch(patch=x)


i = 0
for char in Segmentation.separate_chars(patch=x, coordlist=coord):
    i += 1
    cv2.imwrite("u/B{}.png".format(i), char)
