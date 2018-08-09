import cv2

img_1992 = cv2.imread("ROC_1992.png")
cv2.imshow("1992", img_1992)

img_2014 = cv2.imread("ROC_2014.png")
cv2.imshow("2014", img_2014)

cv2.destroyAllWindows()


