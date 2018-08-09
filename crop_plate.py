import cv2

img = cv2.imread("ROC_1992.png")

with open("boxes_1992.txt") as f:
    lines = f.read().strip().split("\n")

    for (idx, line) in enumerate(lines):
        box = [int(v) for v in line.split()]
        plate = img[box[1]:box[3], box[0]:box[2]]

        cv2.imwrite("plate_%02d.png" % idx, plate)

        cv2.imshow("plate", plate)
        cv2.waitKey()

    cv2.destroyAllWindows()
