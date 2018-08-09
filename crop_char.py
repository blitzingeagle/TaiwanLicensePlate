import cv2

img = cv2.imread("cropped_plates/plate_20.png")

with open("cropped_plates/plate_00.txt") as f:
    lines = f.read().strip().split("\n")

    for (idx, line) in enumerate(lines):
        box = [int(v) for v in line.split()]
        char = img[box[1]:box[3], box[0]:box[2]]

        cv2.imwrite("components/component_%02d.png" % idx, char)

        cv2.imshow("char", char)
        cv2.waitKey()

    cv2.destroyAllWindows()
