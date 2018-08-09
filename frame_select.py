import cv2

# img_1992 = cv2.imread("ROC_1992.png")
img_1992 = cv2.imread("cropped_plates/plate_21.png")
# cv2.imshow("1992", img_1992)

img_2014 = cv2.imread("ROC_2014.png")
# cv2.imshow("2014", img_2014)

(img_1992_h, img_1992_w, _) = img_1992.shape
(x1, y1, x2, y2) = (10, 32, 57, 123)

key = ord('x')
while not key == ord('q'):
    cv2.imshow("img", img_1992[y1:y2, x1:x2])

    key = cv2.waitKey(0) & 0xFF
    if key == ord('w'):
        y1 = max(0, y1-1)
    elif key == ord('s'):
        y1 = min(y2, y1+1)
    elif key == ord('a'):
        x1 = max(0, x1-1)
    elif key == ord('d'):
        x1 = min(x2, x1+1)
    elif key == ord('i'):
        y2 = max(y1, y2-1)
    elif key == ord('k'):
        y2 = min(img_1992_h-1, y2+1)
    elif key == ord('j'):
        x2 = max(x1, x2-1)
    elif key == ord('l'):
        x2 = min(img_1992_w-1, x2+1)
    elif key == ord('t'):
        y1 = max(0, y1-10)
        y2 = max(y1, y2-10)
    elif key == ord('g'):
        y2 = min(img_1992_h-1, y2+10)
        y1 = min(y2, y1+10)
    elif key == ord('f'):
        x1 = max(0, x1-10)
        x2 = max(x1, x2-10)
    elif key == ord('h'):
        x2 = min(img_1992_w-1, x2+10)
        x1 = min(x2, x1+10)

    print(x1, y1, x2, y2)

cv2.destroyAllWindows()


