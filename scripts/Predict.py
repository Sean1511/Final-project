import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
from scripts.ImageSegmetation import image_segment
from scripts.TrainModel import rand_patch
import tensorflow as tf
import glob


FEMALE = 0
MALE = 1
model_name = 'models/vgg16_gender_split_segmentation_19ep.h5'
test_female_path = r'gender_split\test\female'
test_male_path = r'gender_split\test\male'


def predict_image(im_name, model):
    """predict the gender of the writer, extract patches from the image and process them for the model
    :parameter
    im_name: hand writing image name or path to predict gender
    model: use this model to predict image
    :return
    0: if the gender is Female
    1: if the gender is Male
    """
    image = cv2.imread(im_name, 0)
    patches = image_segment(image)
    x_test = []
    for patch in patches:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        patch = cv2.resize(patch, (224, 224))
        patch = tf.keras.applications.vgg16.preprocess_input(patch)
        x_test.append(patch)
    print(im_name + " : ")
    print(len(x_test))
    predictions = model.predict(np.asarray(x_test))

    m_score, f_score = 0, 0
    for prediction in predictions:
        if prediction[0] > 0.5:
            f_score = f_score + 1
        else:
            m_score = m_score + 1

    print('male score : ', m_score)
    print('female score : ', f_score)
    print('classification : {0}'.format('MALE' if m_score > f_score else 'FEMALE'))
    print()
    return MALE if m_score > f_score else FEMALE


if __name__ == '__main__':
    os.chdir(r'F:\לימודים\פרויקט גמר\project')
    model = load_model(model_name)

    n_files = len(glob.glob1(test_female_path, "*.jpg"))
    f_predicted = 0
    for image_name in os.listdir(test_female_path):
        if predict_image(test_female_path + '\\' + image_name, model) == FEMALE:
            f_predicted += 1

    m_predicted = 0
    for image_name in os.listdir(test_male_path):
        if predict_image(test_male_path + '\\' + image_name, model) == MALE:
            m_predicted += 1

    print("summary :")
    print('female predicted : {0} \\ {1}'.format( f_predicted, n_files))
    print('male predicted : {0} \\ {1}'.format(m_predicted, n_files))

    print('Accuracy : {0}%'.format(((f_predicted + m_predicted)/ (n_files * 2)) * 100))


def predict_image2(image, model, preprocess_func, scale):
    """predict the gender of the writer, extract patches from the image and process them for the model
    :parameter
    im_name: hand writing image name or path to predict gender
    model: use this model to predict image
    preprocess_func: the model pre process function to make the patches fit to model
    scale : scale for resize each patch to fit model
    :return
    (float) Gender probability to be male
    (float) Gender probability to be female
    (int) number of patches extracted form the image
    (str) final gender classification: Male or Female
    """
    patches = image_segment(image)   # extract patches from image
    if len(patches) % 2 == 0: patches.append(rand_patch(image)) # if number of patches is even, add 1 random patch to make it odd
    x_test = []
    for patch in patches:   # Process each patch to fit the model
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        patch = cv2.resize(patch, scale)
        patch = preprocess_func(patch)
        x_test.append(patch)

    patches_count = len(x_test)
    print('number of patches extracted from image : ', patches_count)
    predictions = model.predict(np.asarray(x_test))

    m_score, f_score = 0, 0   # count how many patches was classify to each class
    for prediction in predictions:
        if prediction[0] > 0.5:
            f_score = f_score + 1
        else:
            m_score = m_score + 1

    print('male score : ', m_score/patches_count)   # get final classification
    print('female score : ', f_score/patches_count)
    print('classification : {0}'.format('MALE' if m_score > f_score else 'FEMALE'))
    print()
    return m_score/patches_count, f_score/patches_count, patches_count, ('{0}'.format('MALE' if m_score > f_score else 'FEMALE'))


def check_image_size(image):
    """resize image to proper resolution before making patches"""
    y = len(image)  # y is height
    x = len(image[0])  # x is width
    print(x, y)
    if y > 3000 or x > 3000:
        new_x = int(x * (2 / 3))
        new_y = int(y * (2/ 3))
        image = cv2.resize(image, (new_x, new_y))
    elif y > 2500 or x > 2500:
        new_x = int(x * (2.3 / 3))
        new_y = int(y * (2.3 / 3))
        image = cv2.resize(image, (new_x, new_y))
    print(len(image[0]), len(image))
    return image

