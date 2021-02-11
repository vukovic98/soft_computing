import glob
import os

import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import regularizers

import imutils as imutils
import inline as inline
import matplotlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPool2D, AveragePooling2D, MaxPooling3D, Dense, Dropout, Input, Flatten, SeparableConv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras import regularizers, optimizers, losses, models, layers

width = 75
height = 100
channel = 1

img_rows = 100
img_cols = 75

dictionary = {0:'0', 1:'1', 2 :'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A',
    11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K',
    21:'L', 22:'M', 23:'N', 24:'P', 25:'Q', 26:'R', 27:'S', 28:'T', 29:'U',
    30:'V', 31:'W', 32:'X', 33:'Y', 34:'Z'}


def load_data():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        'letter_dataset/train',
        target_size=(img_rows, img_cols),
        batch_size=32,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        'letter_dataset/val',
        target_size=(img_rows, img_cols),
        batch_size=32,
        class_mode='categorical')

    return train_generator, validation_generator

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(35, activation='softmax'))
    # model.summary()

    opt = optimizers.Adam()

    # Compile model
    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=opt, metrics=['accuracy'])

    return model

def resect_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale

    gray = cv2.bilateralFilter(gray, 13, 15, 15)

    edged = cv2.Canny(gray, 30, 200)  # Perform Edge detection

    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    possibleContoures = []

    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            hull = cv2.convexHull(approx, returnPoints=True)
            rectness = rectangleness(hull)
            possibleContoures.append((rectness, approx))

    val = max(possibleContoures, key=lambda item: item[0])[1]

    # Masking the part other than the number plate
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [val], 0, 255, -1, )
    cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped_image = gray[topx:bottomx + 1, topy:bottomy + 1]

    return cropped_image

def process_video(video_path):
    model = create_model()
    print("[ LOADING WEIGHTS... ]")
    model.load_weights("trying_final.h5")
    print("[ WEIGHTS LOADED! ]")

    # ucitavanje videa
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num)  # indeksiranje frejmova
    # analiza videa frejm po frejm
    while True:
        frame_num += 1
        ret_val, frame = cap.read()

        # ako frejm nije zahvacen
        if not ret_val:
            break

        try:
            cropped_image = resect_plate(frame)

            find_characters_contures(cropped_image, model)
            plt.imshow(cropped_image)
            plt.show()
        except:
            continue

    cap.release()
    return True

def find_characters_contures(cropped_image, model):
    real_img = cropped_image.copy()
    cropped_image = cv2.adaptiveThreshold(cropped_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    contours = cv2.findContours(cropped_image.copy(), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    i = 0
    preds = []
    for c in contours:

        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(real_img, (x, y), (x + w, y + h), (122, 122, 232), 1)

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        img_cpy = real_img.copy()
        smoothed = cv2.GaussianBlur(img_cpy, (9, 9), 10)
        img_cpy = cv2.addWeighted(img_cpy, 1.5, smoothed, -0.5, 0)

        # e,img_cpy = cv2.threshold(img_cpy,80,255,cv2.THRESH_BINARY)
        if i != 0:
            mask = np.zeros(img_cpy.shape, np.uint8)
            cv2.drawContours(mask, [approx], 0, 255, -1, )
            cv2.bitwise_and(img_cpy, img_cpy, mask=mask)

            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            img_cpy = img_cpy[topx:bottomx + 1, topy:bottomy + 1]
            img_cpy = cv2.cvtColor(img_cpy,cv2.COLOR_GRAY2RGB)
            plt.imshow(img_cpy)
            plt.show()
            img_cpy = cv2.resize(img_cpy, (img_rows, img_cols))
            img_cpy = np.reshape(img_cpy, (1, img_rows, img_cols, 3))
            prediction = model.predict_classes(img_cpy)
            preds.append(dictionary[prediction[0]])
        else:
            i = i + 1
    for p in preds:
        print(p, end=" ")

    print("---------------------------")
    return cropped_image

def dilate(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((1, 1)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=3)
def rectangleness(hull):
    rect = cv2.boundingRect(hull)
    rectPoints = np.array([[rect[0], rect[1]],
                           [rect[0] + rect[2], rect[1]],
                           [rect[0] + rect[2], rect[1] + rect[3]],
                           [rect[0], rect[1] + rect[3]]])
    intersection_area = cv2.intersectConvexConvex(np.array(rectPoints), hull)[0]
    rect_area = cv2.contourArea(rectPoints)
    rectangleness = intersection_area/rect_area
    return rectangleness

if __name__ == '__main__':
    process_video("./data/vid1.MOV")

    # model = create_model()
    #
    # train_generator, validation_generator = load_data()
    #
    # hist = model.fit_generator(generator=train_generator, validation_data=validation_generator, epochs=8)
    # print("[ TRAINING DONE! ]")
    #
    # # Show results
    # score = model.evaluate_generator(train_generator, verbose=0)
    #
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    #
    # model.save_weights("trying_final.h5")