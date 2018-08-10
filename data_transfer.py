import cv2
import numpy as np
from glob import glob
import os

files = sorted(glob("data/*.jpg"))

for file in files:
    filename = os.path.splitext(os.path.basename(file))[0].upper() + ".png"
    img = cv2.imread(file)
    img = cv2.resize(img, (45, 88))
    cv2.imwrite(os.path.join("generator/2014/characters", filename), img)
