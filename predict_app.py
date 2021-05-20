from tensorflow.keras.models import load_model
import cv2
import base64
import io
from random import randint
import tensorflow as tf
import numpy as np
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)

FEMALE = 0
MALE = 1
current_model = 'Xception'
model = load_model('models/xception_segmentation_split_40ep.h5')
preprocess_func = tf.keras.applications.xception.preprocess_input
scale = (299, 299)


def count_not_white_pixels(im, black_px_min):
    # count how many not white pixels in given image
    black = 0
    for i in range(im.shape[0]):  # traverses through height of the image
        for j in range(im.shape[1]):  # traverses through width of the image
            if im[i][j].all() < 200:
                black += 1
    return black > black_px_min # return true if there is more black pixels than minimum black pixels


def image_segment(image):
    # divide image using 'slide window' to patches ,each patch is 400x400 scale
    height = len(image)  # y is height
    width = len(image[0])  # x is width
    y = 0
    images = []
    while y <= height - 400:
        x = 100
        while x < width - 200:
            div_im = image[y:y+400, x:x+400].copy()
            if count_not_white_pixels(div_im, 5000):
                images.append(div_im)
            x = x + 200
        y = y + 200
    return images


def rand_patch(image):
    # return a random 400x400 scale patch
    y = len(image)  # y is height
    x = len(image[0])  # x is width
    patch = None
    while patch is None:
        ranx = randint(0, x - 400)
        rany = randint(0, y - 400 )
        im = image[rany:rany+400, ranx:ranx+400].copy()
        if count_not_white_pixels(im, 4500):
            patch = im
    return patch


def select_model(name):
    if name == "Xception":
        model = load_model('models/xception_segmentation_split_40ep.h5')
        preprocess_func = tf.keras.applications.xception.preprocess_input
        scale = (299, 299)
    elif name == "VGG16":
        model = load_model('models/vgg16_gender_split_segmetation_19ep.h5')
        preprocess_func = tf.keras.applications.vgg16.preprocess_input
        scale = (224, 224)
    elif name == "Efficienet":
        model = load_model('models/efficientnet_segmentation_split_15ep.h5')
        preprocess_func = tf.keras.applications.efficientnet.preprocess_input
        scale = (224, 224)
    elif name == "Nasnet":
        model = load_model('models/NasNet_segmentation_split_20ep.h5')
        preprocess_func = tf.keras.applications.nasnet.preprocess_input
        scale = (331, 331)
    return model, preprocess_func, scale


def load_default_settings():
    return current_model, model, preprocess_func, scale

def predict_image(image, model, preprocess_func, scale):
    patches = image_segment(image)   # extract patches from image
    if len(patches) % 2 == 0 : patches.append(rand_patch(image)) # if number of patches is even, add 1 random patch to make it odd
    x_test = []
    for patch in patches:   # Process each patch to fit the model
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


@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True) # Get image from user
    encoded = message['image']
    model_name = message['model']
    current_model, model, preprocess_func, scale = load_default_settings()
    if (model_name != current_model):
        model, preprocess_func, scale = select_model(model_name)
        current_model = model_name
    decoded = base64.b64decode(encoded) # decode data
    image_stream = io.BytesIO(decoded)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    prediction = predict_image(image, model, preprocess_func, scale)

    response = {
        'prediction': {
            'Male': prediction[0]*100,
            'Female': prediction[1]*100, 
            'patches_count': prediction[2],
            'classification': prediction[3]
        }
    }

    return jsonify(response)