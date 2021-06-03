import glob
import os
import cv2
from scripts.PreProcess import count_not_white_pixels


def read_data(csv_path, writers_dict):
    """read data and make a dictionary of writers id as keys and gender as values
    :parameter
    csv_path: path to csv file storing the data
    writers_dict: dictionary to store data for use on script
    """
    with open(csv_path, 'r') as csv_file:
        lines = csv_file.readlines()

    for line in lines[1:]:
        data = line.split(',')
        writers_dict[data[0]] = data[1].replace('\n', '')


def remove_empty_rows(img):
    """function to remove blank rows from text image, remove blank rows from the top and bottom of inmage
    :parameter
    img: image to remove blank rows from
    :return
    image with no blank rows
    """
    y = len(img)  # y is height
    x = len(img[0])  # x is width
    h1 = 0
    while not count_not_white_pixels(img[h1:h1 + 100, 0:x], 20):    # remove empty rows from top of the document
        h1 += 100
    h2 = h1
    while count_not_white_pixels(img[h2:h2+250, 0:x], 200) and h2 < y: # remove empty rows from lower part of document
        h2 += 250
    img = img[h1:h2, 0:x].copy()
    return img if h1 != h2 else None


os.chdir(r'F:\לימודים\פרויקט גמר\project')
# train_path = 'train_answers.csv'
test_path = 'test_answers.csv'

# path for directories to save the images sorting by language
m_path_eng = 'ICDAR_test_gender_language/english/male\\'
f_path_eng = 'ICDAR_test_gender_language/english/female\\'

m_path_arb = 'ICDAR_test_gender_language/arabic/male\\'
f_path_arb = 'ICDAR_test_gender_language/arabic/female\\'

writers_gender = {}

# read_data(train_path, writers_gender)
read_data(test_path, writers_gender)

directory = 'ICDAR_dataset\\'
n_files = len(glob.glob1(directory, "*.jpg"))
i = 0


for filename in os.listdir(directory):
    i += 1
    print('\r Processing {0} / {1}'.format(i, n_files), end='')

    if filename.endswith(".jpg"):
        id = str(int(filename.split('_')[0]))  # get writer id of specific image
        language = int(filename.split('_')[1].split('.')[0])
        image = cv2.imread(directory + filename)
        if type(image) is None: # if image reading is failed, print image name to screen and break loop
            print('0 ' + filename)
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   # image to gray scale
        txt_im = remove_empty_rows(image)    # remove blank rows from image
        if txt_im is not None: image = txt_im

        if writers_gender[id] == '1':    # if the writer is male
            if language <= 2:    # if language is arabic
                cv2.imwrite(m_path_arb + filename, image)
            else:    # language is english
                cv2.imwrite(m_path_eng + filename, image)
        else:        # writer is female
            if language <= 2:     # if language is arabic
                cv2.imwrite(f_path_arb + filename, image)
            else:                 # language is english
                cv2.imwrite(f_path_eng + filename, image)

