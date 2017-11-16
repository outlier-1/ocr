import cv2
import matlab.engine

eng = matlab.engine.start_matlab()
org_rect = cv2.imread("../Text_Detection/rectangles/E6CV.png")
path = eng.binarizeImage("../Text_Detection/rectangles/E6CV.png", "tmp/tmp_rect.png")
tmp_rect = cv2.imread(path)
line_coordinates = []
pointer_X = 0
y_one, y_two = 0, tmp_rect.shape[0] - 1
x_one, x_two = None, None
control = False
while pointer_X < tmp_rect.shape[1]:
    # Look for first occur
    pointer_X += 1
    control = False
    # Look for character without space,
    if 0 in tmp_rect[:, pointer_X - 1]:
        x_one = pointer_X - 1
        x_two = x_one

        # Check for last 10 pixels, if line drew, don't draw new line, otherwise draw.
        for a in range(1, 10):
            if [x_one-a, 0, x_one-a, y_two] in line_coordinates:
                control = True
        if control is False:
            line_coordinates.append([x_one, y_one, x_two, y_two])

        # Look for end of the character
        while True:
            pointer_X += 1
            # Draw a line to the end of the character.
            if 0 not in tmp_rect[:, pointer_X]:
                x_one = pointer_X + 1
                x_two = x_one
                line_coordinates.append([x_one, y_one, x_two, y_two])
                break

# Draw Lines to original image.
for item in line_coordinates:
    cv2.line(org_rect, (item[0], item[1]), (item[2], item[3]), (0, 255, 0), 1)

cv2.imshow("bitti", org_rect)
cv2.waitKey(1000000)
