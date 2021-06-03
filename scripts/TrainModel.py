import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D,  \
                                    AveragePooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import glob
from tensorflow.keras.models import load_model
import os
import random
from random import randint
from scripts.ImageSegmetation import image_segment
from scripts.PreProcess import count_not_white_pixels
from scripts.Predict import rand_patch


FEMALE = 0
MALE = 1
SCALE = (112, 112)
model_name = r'models/simple_model_quwi_english_98ep_32units_112x112.h5'


fpath_train = 'QUWI_subset_segmentation/english/train/female'
fpath_test = 'QUWI_subset_segmentation/english/test/female'
fpath_valid = 'QUWI_subset_segmentation/english/valid/female'

mpath_train = 'QUWI_subset_segmentation/english/train/male'
mpath_test = 'QUWI_subset_segmentation/english/test/male'
mpath_valid = 'QUWI_subset_segmentation/english/valid/male'


def read_images(path, images_list, labels_list, label):
    """read images and the correct labels
    :parameter
    path: path of directory to read from
    images_list : list of images to append the images from path directory
    labels_list : list of labels to append the correct label from path directory
    label : the gender of the writer of the image
    """
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            image = cv2.imread(path + '\\' + filename)
            image = cv2.resize(image, SCALE)  # Resize image to model input scale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image, axis=2)
            images_list.append(image)
            labels_list.append(label)


def save_model(model, model_name):
    """Save model"""
    model.save(model_name)
    print('Model ' + model_name + ' successfully saved')


def shuffle_2_lists(a, b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return a, b


def predict_image(im_name, model):
    """identify by the model if handwriting image wrote by male or female"""
    image = cv2.imread(im_name, 0)  # read image as gray scale

    patches = image_segment(image)   # extract patches from image
    if len(patches) % 2 == 0:
        patches.append(rand_patch(image))  # if number of patches is even, add 1 random patch to make it odd
    print(im_name + " : ")
    x_test = []
    for patch in patches:   # Process each patch to fit the model
        patch = cv2.resize(patch, SCALE)  # Resize image to model input scale
        # patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch = np.expand_dims(patch, axis=2)
        x_test.append(patch)

    x_test = np.asarray(x_test)
    x_test = x_test.astype('float32')
    x_test /= 255

    patches_count = len(x_test)
    print('number of patches extracted from image : ', patches_count)
    predictions = model.predict(np.asarray(x_test))

    m_score, f_score = 0, 0   # count how many patches was classify to each class
    for prediction in predictions:
        if prediction < 0.5:
            f_score = f_score + 1
        else:
            m_score = m_score + 1

    print('male score : ', m_score)   # get final classification
    print('female score : ', f_score)
    print('classification : {0}'.format('MALE' if m_score > f_score else 'FEMALE'))
    print()
    return MALE if m_score > f_score else FEMALE


def get_test_results(test_female_path, test_male_path):
    """ calculate the accuracy of the model
    :parameter
    test_female_path : path to female handwriting folder
    test_male_path : path to male handwriting folder
    """
    model = load_model(model_name)
    n_files = len(glob.glob1(test_female_path, "*.jpg"))
    f_predicted = 0
    for image_name in os.listdir(test_female_path):
        if predict_image(test_female_path + '/' + image_name, model) == FEMALE:
            f_predicted += 1

    m_predicted = 0
    for image_name in os.listdir(test_male_path):
        if predict_image(test_male_path + '/' + image_name, model) == MALE:
            m_predicted += 1

    print("summary :")
    print('female predicted : {0} \\ {1}'.format(f_predicted, n_files))
    print('male predicted : {0} \\ {1}'.format(m_predicted, n_files))

    print('Accuracy : {0}%'.format(((f_predicted + m_predicted) / (n_files * 2)) * 100))


def train_model():
    """pre process the images and the labels and train the model"""
    x_train = []
    x_test = []
    x_valid = []

    y_train = []
    y_test = []
    y_valid = []

    # reading train
    read_images(fpath_train, x_train, y_train, FEMALE)
    read_images(mpath_train, x_train, y_train, MALE)

    # reading test
    read_images(fpath_test, x_test, y_test, FEMALE)
    read_images(mpath_test, x_test, y_test, MALE)

    # reading valid
    read_images(fpath_valid, x_valid, y_valid, FEMALE)
    read_images(mpath_valid, x_valid, y_valid, MALE)

    print('x_train - ', len(x_train))
    print('y_train - ', len(y_train))

    print('x_test - ', len(x_test))
    print('y_test - ', len(y_test))

    print('x_valid - ', len(x_valid))
    print('y_valid - ', len(y_valid))

    # x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.10, random_state=42)
    x_train = np.asarray(x_train)
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = np.asarray(y_train)
    # x_train, y_train = shuffle_2_lists(x_train, y_train)

    x_test = np.asarray(x_test)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = np.asarray(y_test)

    x_valid = np.asarray(x_valid)
    x_valid = x_valid.astype('float32')
    x_valid /= 255
    y_valid = np.asarray(y_valid)
    # x_valid, y_valid = shuffle_2_lists(x_valid, y_valid)

    # build model
    model = Sequential()

    model.add(Conv2D(32, (5, 5), strides=(1, 1), name = 'conv0', input_shape=(112, 112, 1)))

    model.add(BatchNormalization(axis=3, name = 'bn0'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), name='max_pool'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), name="conv1"))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((3, 3), name='avg_pool'))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(300, activation="relu", name='rl'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', name='sm'))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-4),
                  metrics=['accuracy'])

    batch_size = 32
    nb_epoch = 98

    # Train model
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=nb_epoch,
                        validation_data=(x_valid, y_valid),
                        shuffle=True,
                        )

    loss, accuracy = model.evaluate(x_test, y_test)
    print('loss = ', loss, 'accuracy = ', accuracy)
    save_model(model, model_name)


if __name__ == "__main__":

    os.chdir(r'F:\לימודים\פרויקט גמר\project')
    test_f_path = r'QUWI_subset\english\test\female'
    test_m_path = r'QUWI_subset\english\test\male'

    # build and train model
    # train_model()

    # load and test model
    # get_test_results(test_female_path=test_f_path, test_male_path=test_m_path)

    test_f_path = r'QUWI_subset\arabic\test\female'
    test_m_path = r'QUWI_subset\arabic\test\male'

    get_test_results(test_female_path=test_f_path, test_male_path=test_m_path)