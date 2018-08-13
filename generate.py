import cv2
import numpy as np
import os
from random import randint
import re
import string

plate_height_1992 = 146
plate_width_1992 = 308

plate_height_2014 = 155
plate_width_2014 = 365

regions = ["gaoxiongshi", "jinmenxian", "lianjiangxian", "taibeishi", "taiwansheng"]


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

    baseplate = cv2.resize(baseplate, (plate_width_1992, plate_height_1992))
    return baseplate


def generate_2014(template, characters):
    generator_dir = "generator/2014"
    template_basename = os.path.join(generator_dir, "templates", template)
    with open(template_basename + ".txt") as f:
        lines = f.read().strip().split("\n")
        lines = [[int(v) for v in line.split()] for line in lines]
        fgcolor = np.array(lines[0])
        bgcolor = np.array(lines[1])
        boxes = lines[2:]

    baseplate = cv2.imread(template_basename + ".png")
    characters = characters.replace('-', '')
    img_paths = ["generator/2014/characters/%s.png" % c for c in characters]

    for (idx, box) in enumerate(boxes):
        (width, height) = (box[2] - box[0], box[3] - box[1])
        feature = cv2.bitwise_not(cv2.imread(img_paths[idx]))
        feature = cv2.resize(feature, (width, height)).astype(np.float32) / 255.
        component = feature * (fgcolor - bgcolor) + bgcolor
        baseplate[box[1]:box[3], box[0]:box[2]] = component

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


def generate_random_plates(format, I_O_removed=True, count=1):
    plates = set()

    while len(plates) < count:
        plates.add(generate_random_plate(format, I_O_removed=I_O_removed))

    return plates


def demo1992():
    regions = ["gaoxiongshi", "jinmenxian", "lianjiangxian", "taibeishi", "taiwansheng", "blank"]

    (rows, cols) = (5, 3)
    canvas = np.zeros(shape=(rows * plate_height_1992, cols * plate_width_1992, 3), dtype=np.uint8)
    plates = []

    region = regions[randint(0, len(regions) - 2)]
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

    region = regions[randint(0, len(regions) - 2)]
    plate_code = generate_random_plate("AA-XX")
    plate = generate_1992("white_green_2_2", plate_code, region)
    plates.append(plate)

    region = regions[randint(0, len(regions) - 2)]
    plate_code = generate_random_plate("AA-XXX")
    plate = generate_1992("red_white_2_3", plate_code, region)
    plates.append(plate)

    region = regions[randint(0, len(regions) - 2)]
    plate_code = generate_random_plate("XXX-AA")
    plate = generate_1992("white_green_3_2", plate_code, region)
    plates.append(plate)

    region = regions[randint(0, len(regions) - 2)]
    plate_code = generate_random_plate("XXX-AA")
    plate = generate_1992("green_white_3_2", plate_code, region)
    plates.append(plate)

    region = regions[randint(0, len(regions) - 2)]
    plate_code = generate_random_plate("XXX-AA")
    plate = generate_1992("white_red_3_2", plate_code, region)
    plates.append(plate)

    region = regions[randint(0, len(regions) - 2)]
    plate_code = generate_random_plate("A?-XXX")
    plate = generate_1992("white_red_2_3", plate_code, region)
    plates.append(plate)

    region = regions[randint(0, len(regions) - 2)]
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
        (y1, x1) = (row * plate_height_1992, col * plate_width_1992)
        (y2, x2) = (y1 + plate_height_1992, x1 + plate_width_1992)
        canvas[y1:y2, x1:x2] = plate

    return canvas


def demo2014():
    templates = ["black_white_2014", "black_yellow_2014", "electric_black_white_2014", "electric_green_white_2014",
                 "electric_red_white_2014", "electric_handicap_2014", "green_white_2014", "handicap_2014",
                 "red_white_2014", "white_green_2014", "white_red_2014"]

    (rows, cols) = (5, 3)
    canvas = np.zeros(shape=(rows * plate_height_2014, cols * plate_width_2014, 3), dtype=np.uint8)
    plates = []

    for template in templates:
        plate_code = generate_random_plate("AAA-XXXX")
        plate = generate_2014(template, plate_code)
        plates.append(plate)

    for (idx, plate) in enumerate(plates):
        (row, col) = (idx % rows, idx // rows)
        (y1, x1) = (row * plate_height_2014, col * plate_width_2014)
        (y2, x2) = (y1 + plate_height_2014, x1 + plate_width_2014)
        print(plate.shape)
        canvas[y1:y2, x1:x2] = plate

    return canvas


def create_dataset():
    regions = ["gaoxiongshi", "jinmenxian", "lianjiangxian", "taibeishi", "taiwansheng", "blank"]
    fs_template = "trainB/%s_%s.png"

    plate_codes = generate_random_plates("??-XXXX", count=1000)
    for plate_code in plate_codes:
        region = regions[randint(0, len(regions) - 2)]
        plate = generate_1992("black_white_2_4", plate_code, region)
        cv2.imwrite(fs_template % (plate_code, "black_white_2_4"), plate)

    plate_codes = generate_random_plates("XXXX-??", count=1000)
    for plate_code in plate_codes:
        region = regions[randint(0, len(regions) - 1)]
        plate = generate_1992("black_white_4_2", plate_code, region)
        cv2.imwrite(fs_template % (plate_code, "black_white_4_2"), plate)

    plate_codes = generate_random_plates("??-??", count=250)
    for plate_code in plate_codes:
        region = regions[randint(0, len(regions) - 1)]
        plate = generate_1992("green_white_2_2", plate_code, region)
        cv2.imwrite(fs_template % (plate_code, "green_white_2_2"), plate)

    plate_codes = generate_random_plates("AA-XX", count=250)
    for plate_code in plate_codes:
        region = regions[randint(0, len(regions) - 2)]
        plate = generate_1992("white_green_2_2", plate_code, region)
        cv2.imwrite(fs_template % (plate_code, "white_green_2_2"), plate)

    plate_codes = generate_random_plates("AA-XXX", count=500)
    for plate_code in plate_codes:
        region = regions[randint(0, len(regions) - 2)]
        plate = generate_1992("red_white_2_3", plate_code, region)
        cv2.imwrite(fs_template % (plate_code, "red_white_2_3"), plate)

    plate_codes = generate_random_plates("XXX-AA", count=500)
    for plate_code in plate_codes:
        region = regions[randint(0, len(regions) - 2)]
        plate = generate_1992("white_green_3_2", plate_code, region)
        cv2.imwrite(fs_template % (plate_code, "white_green_3_2"), plate)

    plate_codes = generate_random_plates("XXX-AA", count=500)
    for plate_code in plate_codes:
        region = regions[randint(0, len(regions) - 2)]
        plate = generate_1992("green_white_3_2", plate_code, region)
        cv2.imwrite(fs_template % (plate_code, "green_white_3_2"), plate)

    plate_codes = generate_random_plates("XXX-AA", count=250)
    for plate_code in plate_codes:
        region = regions[randint(0, len(regions) - 2)]
        plate = generate_1992("white_red_3_2", plate_code, region)
        cv2.imwrite(fs_template % (plate_code, "white_red_3_2"), plate)

    plate_codes = generate_random_plates("A?-XXX", count=250)
    for plate_code in plate_codes:
        region = regions[randint(0, len(regions) - 2)]
        plate = generate_1992("white_red_2_3", plate_code, region)
        cv2.imwrite(fs_template % (plate_code, "white_red_2_3"), plate)

    plate_codes = generate_random_plates("AA-XXX", count=250)
    for plate_code in plate_codes:
        region = regions[randint(0, len(regions) - 2)]
        plate = generate_1992("black_yellow_2_3", plate_code, region)
        cv2.imwrite(fs_template % (plate_code, "black_yellow_2_3"), plate)

    plate_codes = generate_random_plates("XXXX-??", count=500)
    for plate_code in plate_codes:
        region = None
        plate = generate_1992("handicap_4_2", plate_code, region)
        cv2.imwrite(fs_template % (plate_code, "handicap_4_2"), plate)

    plate_codes = generate_random_plates("XXX-A?", count=250)
    for plate_code in plate_codes:
        region = None
        plate = generate_1992("electric_red_white_3_2", plate_code, region)
        cv2.imwrite(fs_template % (plate_code, "electric_red_white_3_2"), plate)

    plate_codes = generate_random_plates("XXXX-A?", count=250)
    for plate_code in plate_codes:
        region = None
        plate = generate_1992("electric_black_white_4_2", plate_code, region)
        cv2.imwrite(fs_template % (plate_code, "electric_black_white_4_2"), plate)

    plate_codes = generate_random_plates("XXX-A?", count=250)
    for plate_code in plate_codes:
        region = None
        plate = generate_1992("electric_green_white_3_2", plate_code, region)
        cv2.imwrite(fs_template % (plate_code, "electric_green_white_3_2"), plate)


if __name__ == "__main__":
    # create_dataset()
    #
    # templates = ["black_white_2014", "black_yellow_2014", "electric_black_white_2014", "electric_green_white_2014",
    #              "electric_red_white_2014", "green_white_2014", "handicap_2014", "red_white_2014", "white_green_2014",
    #              "white_red_2014"]
    # counts = [2200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
    # fs_template = "trainB/%s_%s.png"
    #
    # for (idx, template) in enumerate(templates):
    #     plate_codes = generate_random_plates("AAA-XXXX", count=counts[idx])
    #     for plate_code in plate_codes:
    #         plate = generate_2014(template, plate_code)
    #         cv2.imwrite(fs_template % (plate_code, template), plate)

    img1 = demo1992()
    img2 = demo2014()

    cv2.imshow("1992", img1)
    cv2.imshow("2014", img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
