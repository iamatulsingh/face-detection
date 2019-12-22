import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model

import cv2

from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions

from model import get_model

model = get_model()
# print(model.summary())


def extract_face(filename, required_size=(224, 224)):
    img = cv2.imread(filename)
	# create the detector, using default weights
    detector = MTCNN()
	# detect faces in the image
    results = detector.detect_faces(img)
    x, y, width, height = results[0]['box']
    face = img[y:y+height, x:x+width]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array, face

# @tf.function
def who_is_this(img, vgg_face_descriptor):
    face_array, face = extract_face(img)
    face_array = face_array.astype('float32')
    input_sample = np.expand_dims(face_array, axis=0)
    img_prediction = vgg_face_descriptor.predict(preprocess_input(input_sample))
    results = decode_predictions(img_prediction)
    prediction = results[0][0][0].replace("b'", "").replace("'","")
    
    # plt.imshow(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))
    # plt.title(results[0][0][0].replace("b'", "").replace("'",""))
    
    # plt.figure(figsize=(12, 12))
    # ax1 = plt.subplot(121)
    # ax2 = plt.subplot(122)

    # ax1.imshow(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))
    # ax1.set_title(results[0][0][0].replace("b'", "").replace("'","") \
    #              + " original image".upper(),\
    #                 fontdict={'fontsize': 18, 'fontweight': 'bold'})
    # ax2.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    # ax2.set_title(results[0][0][0].replace("b'", "").replace("'","")\
    #             + " face".upper(),\
    #             fontdict={'fontsize': 18, 'fontweight': 'bold'})

    # plt.show()

    return prediction

# @tf.function
def get_prediction(image_path):
    model.load_weights(os.path.join(os.getcwd(), "weight", "vgg_face_weights.h5"))
    vgg_face_descriptor = Model(inputs=model.layers[0].input,\
                                outputs=model.layers[-2].output)

    # test_folder = os.path.join(os.getcwd(), "test")
    # who_is_this(os.path.join(test_folder, "Amir-Khan.jpg"))
    return who_is_this(image_path, vgg_face_descriptor)

if __name__ == "__main__":
    print(get_prediction('test/Amir-Khan.jpg'))