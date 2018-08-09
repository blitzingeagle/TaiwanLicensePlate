import cv2
from glob import glob
import numpy as np
import os
from random import randint
import re
import string

plate_height = 146
plate_width = 308


def generate_1992(template, characters, region=None):
    generator_dir = "generator/1992"
    template_basename = os.path.join(generator_dir, "templates", template)
    with open(template_basename + ".txt") as f:
        lines = f.read().strip().split("\n")
        lines = [[int(v) for v in line.split()] for line in lines]
        fgcolor = np.array(lines[0])
        bgcolor = np.array(lines[1])
        boxes = lines[2:]

    baseplate = cv2.imread(template_basename + ".png")
    characters = characters.replace('-', '')
    img_paths = ["generator/1992/region/%s.png" % region] + ["generator/1992/characters/%s.png" % c for c in characters]
    if region is None:
        del img_paths[0]

    for (idx, box) in enumerate(boxes):
        (width, height) = (box[2] - box[0], box[3] - box[1])
        feature = cv2.bitwise_not(cv2.imread(img_paths[idx]))
        feature = cv2.resize(feature, (width, height)).astype(np.float32) / 255.
        component = feature * (fgcolor-bgcolor) + bgcolor
        baseplate[box[1]:box[3], box[0]:box[2]] = component

    baseplate = cv2.resize(baseplate, (plate_width, plate_height))
    return baseplate


def generate_random_plate(format, I_O_removed=True):
    '''
    Generates a random license plate number using the provided format.
    :param format: string containing 'A', 'X', and '?'. 'A' for letters, 'X' for numbers, 'E' for either.
    :param I_O_removed: set to false if license plate can contain 'I' and 'O'
    :return: string of the license plate number
    '''
    letters = string.ascii_uppercase
    numbers = string.digits

    if I_O_removed:
        letters = re.sub("[IO]", "", letters)

    plate = ""
    for idx in range(len(format)):
        c = format[idx]
        if c == "A":
            plate += letters[randint(0, len(letters)-1)]
        elif c == "X":
            plate += numbers[randint(0, len(numbers)-1)]
        elif c == "?":
            r = randint(0, len(letters) + len(numbers) - 1)
            plate += letters[r] if r < len(letters) else numbers[r - len(letters)]
        else:
            plate += c

    return plate


def demo():
    (rows, cols) = (5, 3)
    canvas = np.zeros(shape=(rows * plate_height, cols * plate_width, 3), dtype=np.uint8)
    plates = []

    region = regions[randint(0, len(regions) - 1)]
    plate_code = generate_random_plate("??-XXXX")
    plate = generate_1992("black_white_2_4", plate_code, region)
    plates.append(plate)

    region = regions[randint(0, len(regions) - 1)]
    plate_code = generate_random_plate("XXXX-??")
    plate = generate_1992("black_white_4_2", plate_code, region)
    plates.append(plate)

    region = regions[randint(0, len(regions) - 1)]
    plate_code = generate_random_plate("??-??")
    plate = generate_1992("green_white_2_2", plate_code, region)
    plates.append(plate)

    region = regions[randint(0, len(regions) - 1)]
    plate_code = generate_random_plate("AA-XX")
    plate = generate_1992("white_green_2_2", plate_code, region)
    plates.append(plate)

    region = regions[randint(0, len(regions) - 1)]
    plate_code = generate_random_plate("AA-XXX")
    plate = generate_1992("red_white_2_3", plate_code, region)
    plates.append(plate)

    region = regions[randint(0, len(regions) - 1)]
    plate_code = generate_random_plate("XXX-AA")
    plate = generate_1992("white_green_3_2", plate_code, region)
    plates.append(plate)

    region = regions[randint(0, len(regions) - 1)]
    plate_code = generate_random_plate("XXX-AA")
    plate = generate_1992("green_white_3_2", plate_code, region)
    plates.append(plate)

    region = regions[randint(0, len(regions) - 1)]
    plate_code = generate_random_plate("XXX-AA")
    plate = generate_1992("white_red_3_2", plate_code, region)
    plates.append(plate)

    region = regions[randint(0, len(regions) - 1)]
    plate_code = generate_random_plate("A?-XXX")
    plate = generate_1992("white_red_2_3", plate_code, region)
    plates.append(plate)

    region = regions[randint(0, len(regions) - 1)]
    plate_code = generate_random_plate("AA-XXX")
    plate = generate_1992("black_yellow_2_3", plate_code, region)
    plates.append(plate)

    region = None
    plate_code = generate_random_plate("XXXX-??")
    plate = generate_1992("handicap_4_2", plate_code, region)
    plates.append(plate)

    region = None
    plate_code = generate_random_plate("XXX-A?")
    plate = generate_1992("electric_red_white_3_2", plate_code, region)
    plates.append(plate)

    region = None
    plate_code = generate_random_plate("XXXX-A?")
    plate = generate_1992("electric_black_white_4_2", plate_code, region)
    plates.append(plate)

    region = None
    plate_code = generate_random_plate("XXX-A?")
    plate = generate_1992("electric_green_white_3_2", plate_code, region)
    plates.append(plate)

    for (idx, plate) in enumerate(plates):
        (row, col) = (idx % rows, idx // rows)
        (y1, x1) = (row * plate_height, col * plate_width)
        (y2, x2) = (y1 + plate_height, x1 + plate_width)
        canvas[y1:y2, x1:x2] = plate

    return canvas


if __name__ == "__main__":
    regions = ["gaoxiongshi", "jinmenxian", "lianjiangxian", "taibeishi", "taiwansheng"]

    for x in range(1000):
        canvas = demo()
        cv2.imshow("demo", canvas)
        cv2.waitKey(5)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # generate_func = [generate_AA_XXXX, generate_AX_XXXX, generate_XA_XXXX, generate_XXXX_AA, generate_green_text_2_2,
    #                  generate_green_plate_2_2, generate_red_text_2_3, generate_green_text_3_2, generate_green_plate_3_2,
    #                  generate_red_plate_2_3]
    # for x in range(10000):
    #     generate = generate_func[randint(0, len(generate_func) - 1)]
    #     img = generate()
    #     cv2.imshow("generated", img)
    #     cv2.waitKey(500)
    # cv2.destroyAllWindows()
