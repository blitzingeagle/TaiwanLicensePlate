import cv2
import numpy as np


def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

img = cv2.imread("components/component_04.png", cv2.IMREAD_GRAYSCALE)
out = cv2.resize(im2double(img), (48, 91))
cv2.imshow("blacked", out)
cv2.imwrite("blacked.png", out*255)

cv2.waitKey()
cv2.destroyAllWindows()

