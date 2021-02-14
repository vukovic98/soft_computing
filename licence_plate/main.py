import pytesseract as pytesseract
import imutils as imutils
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, losses, models, layers
import hog_svm_train as hog_train

width = 75
height = 100
channel = 1

img_rows = 100
img_cols = 75

dictionary = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A',
              11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K',
              21: 'L', 22: 'M', 23: 'N', 24: 'P', 25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U',
              30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z', 35: '-'}

resultDictionary = {}


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


def customThrash(image):
    ret, img = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    return img


def load_data():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        'letter_dataset/train',
        target_size=(img_rows, img_cols),
        batch_size=32,
        color_mode='grayscale',
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        'letter_dataset/val',
        target_size=(img_rows, img_cols),
        batch_size=32,
        color_mode='grayscale',
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
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(36, activation='softmax'))
    # model.summary()

    opt = optimizers.Adam()

    # Compile model
    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=opt, metrics=['accuracy'])

    return model


def resect_plate(img, hog, svm):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale

    # plt.imshow(img)
    # plt.show()

    gray = cv2.bilateralFilter(gray, 13, 15, 15)

    # plt.imshow(gray)
    # plt.show()

    lower_white = np.array([95, 95, 95], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)

    thresh1 = cv2.inRange(img, lower_white, upper_white)

    # plt.imshow(thresh1)
    # plt.show()

    edged = cv2.Canny(thresh1, 30, 200)  # Perform Edge detection

    dilated = dilate(edged, 7)

    plt.imshow(dilated, cmap="gray")
    plt.show()

    # plt.imshow(edged)
    # plt.show()

    contours = cv2.findContours(dilated.copy(), cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    possibleContoures = []
    hogContours = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Draw the rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        plt.imshow(img)
        plt.show()
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        print(len(approx))

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        # if len(approx) == 4:
        hull = cv2.convexHull(approx, returnPoints=True)
        rectness = rectangleness(hull)

        if w > h:
            possibleContoures.append((rectness, approx))
            hogContours.append((c, approx))

    if len(possibleContoures) == 0:
        return None

    hogResults = []

    for c in hogContours:
        x, y, w, h = cv2.boundingRect(c[0])
        window = img[y:y + h, x:x + w]
        # plt.imshow(window)
        # plt.show()
        print("STARTED PROCESSING")
        score = hog_train.process_image(window, hog, svm)
        hogResults.append((c, score))
        print("Score: " + str(score))
        print("END OF PROCESSING")

    val = max(hogResults, key=lambda item: item[1])[0][1]

    # Masking the part other than the number plate
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [val], 0, 255, -1, )
    cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped_image = gray[topx:bottomx, topy:bottomy]

    ret, thresh = cv2.threshold(cropped_image, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if (len(areas) != 0):
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        x, y, w, h = cv2.boundingRect(cnt)
        bounds = cv2.boundingRect(cnt)
        cv2.rectangle(cropped_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        secondCrop = cropped_image[y:y + h, x:x + w]
    else:
        secondCrop = cropped_image

    plt.imshow(cropped_image)
    plt.show()
    return secondCrop


alphabet = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "Ć", "Č", "C", "D", "Đ", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "R", "Š",
            "S", "T", "U", "V", "W", "X", "Y", "Ž", "Z"]


def process_video(video_path, hog, svm):
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
        # plt.imshow(frame)
        # plt.show()
        # ako frejm nije zahvacen
        if not ret_val:
            break

        cropped_image = resect_plate(frame, hog, svm)
        if cropped_image is None:
            print("Failed to find plate on this frame.")
        else:
            # plt.imshow(cropped_image)
            # plt.show()
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
            print(pytesseract.image_to_string(cropped_image))
            find_characters_contures(cropped_image, model)

    cap.release()
    return True


def find_characters_contures(cropped_image, model):
    real_img = cropped_image.copy()

    cropped_image = cv2.adaptiveThreshold(cropped_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # plt.imshow(cropped_image)
    # plt.show()

    contours = cv2.findContours(cropped_image.copy(), cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    contours.remove(contours[0])

    i = 0
    rejected = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > h:
            i = i + 1
            continue
        for c1 in contours:
            x1, y1, w1, h1 = cv2.boundingRect(c1)
            if w1 > h1:
                continue
            if np.array_equal(c, c1):
                print("equal")
                continue

            if x1 < x + w / 2 < x1 + w1 and y1 < y + h / 2 < y1 + h1:
                print(str(x + w / 2) + ", " + str(y + h / 2) + ", " + str(x1) + ", " + str(x1 + w1) + ", " + str(
                    y1) + ", " + str(y1 + h1))
                if w < w1:
                    if i not in rejected:
                        rejected.append(i)
        i = i + 1

    print(rejected)

    for index in reversed(rejected):
        if (index < len(contours)):
            contours = np.delete(contours, index)
    i = 0
    preds = []

    to_predict = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > h:
            continue
        print(w, h)
        cv2.rectangle(real_img, (x, y), (x + w, y + h), (122, 122, 232), 1)

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        img_cpy = real_img.copy()
        smoothed = cv2.GaussianBlur(img_cpy, (9, 9), 10)
        img_cpy = cv2.addWeighted(img_cpy, 1.5, smoothed, -0.5, 0)

        e, img_cpy = cv2.threshold(img_cpy, 80, 255, cv2.THRESH_BINARY)
        # plt.imshow(real_img)
        # plt.show()
        # if i != 0:
        mask = np.zeros(img_cpy.shape, np.uint8)
        cv2.drawContours(mask, [approx], 0, 255, -1, )
        cv2.bitwise_and(img_cpy, img_cpy, mask=mask)
        # plt.imshow(img_cpy)
        # plt.show()
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        img_cpy = img_cpy[topx:bottomx + 1, topy:bottomy + 1]
        # img_cpy = cv2.cvtColor(img_cpy,cv2.COLOR_GRAY2RGB)
        # plt.imshow(img_cpy)
        # plt.show()
        img_cpy = erode(img_cpy, 1)
        # plt.imshow(img_cpy)
        # plt.show()
        img_cpy = cv2.resize(img_cpy, (img_cols, img_rows))
        img_cpy = np.reshape(img_cpy, (1, img_rows, img_cols, 1))
        to_predict.append(img_cpy)

        prediction = model.predict(img_cpy)

        # plt.imshow(img_cpy)
        # plt.show()
        try:
            print(dictionary[np.where(prediction[0] == 1)[0][0]])
            preds.append(dictionary[np.where(prediction[0] == 1)[0][0]])
        except:
            continue
        # else:
        #   i = i + 1

    # input = prepare_for_ann(to_predict)
    # result=model.predict(np.array(input, np.float32))
    # print(result[0])

    result = ""

    for p in preds:
        result = result + p
        print(p, end=" ")

    if result in resultDictionary:
        resultDictionary[result] += 1
    else:
        if result != '':
            resultDictionary[result] = 1

    print(resultDictionary)

    print("---------------------------")
    return cropped_image


def dilate(image, iter):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=iter)


def dilate22(image, iter):
    kernel = np.ones((2, 2))  # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=iter)


def erode(image, iter):
    kernel = np.ones((2, 2))  # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=iter)


def rectangleness(hull):
    rect = cv2.boundingRect(hull)
    rectPoints = np.array([[rect[0], rect[1]],
                           [rect[0] + rect[2], rect[1]],
                           [rect[0] + rect[2], rect[1] + rect[3]],
                           [rect[0], rect[1] + rect[3]]])
    intersection_area = cv2.intersectConvexConvex(np.array(rectPoints), hull)[0]
    rect_area = cv2.contourArea(rectPoints)
    rectangleness = intersection_area / rect_area
    return rectangleness


if __name__ == '__main__':
    hog, svm = hog_train.trainHOG()

    process_video("./data/BP005TI.mp4", hog, svm)

    print("Najverovatniji rezultat je: \n----------\n" + max(resultDictionary,
                                                             key=resultDictionary.get) + "\n----------\nPronadjen je u " + str(
        resultDictionary[max(resultDictionary, key=resultDictionary.get)]) + " frejmova.")

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
