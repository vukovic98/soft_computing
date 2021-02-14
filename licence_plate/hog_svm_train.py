import os
import numpy as np
import cv2 # OpenCV
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def trainHOG():

    train_dir = 'license_plate_train_dataset/'
    hog, x_train, x_test, y_train, y_test = hog_desc(train_dir, 140, 60)
    clf_svm = svm_classifier(x_train, x_test, y_train, y_test)
    return hog, clf_svm

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def display_image(image):
    plt.imshow(image, 'gray')
    plt.show()

# train_dir = 'vehiclel_license_plate_train_dataset/'

def hog_desc(train_dir, x, y):
    list_dir = os.listdir(train_dir)

    pos_imgs = []
    neg_imgs = []

    for img_name in list_dir:
        img_path = os.path.join(train_dir, img_name)
        img = load_image(img_path)
        img.resize(y,x)
        if 'pos' in img_name:
            pos_imgs.append(img)
        elif 'neg' in img_name:
            neg_imgs.append(img)   

    print("Positive images #: ", len(pos_imgs))
    print("Negative images #: ", len(neg_imgs))


    pos_features = []
    neg_features = []
    labels = []

    nbins = 9 # broj binova
    cell_size = (8, 8) # broj piksela po celiji
    block_size = (3, 3) # broj celija po bloku

    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                    img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    for img in pos_imgs:
        pos_features.append(hog.compute(img))
        labels.append(1)

    for img in neg_imgs:
        neg_features.append(hog.compute(img))
        labels.append(0)

    pos_features = np.array(pos_features)
    neg_features = np.array(neg_features)
    x = np.vstack((pos_features, neg_features))
    y = np.array(labels)

    #podjela trening skupa na train i validacioni
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

    return hog, x_train, x_test, y_train, y_test

# transformisemo u oblik pogodan za scikit-learn
def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))

def svm_classifier(x_train, x_test, y_train, y_test):
    x_train = reshape_data(x_train)
    x_test = reshape_data(x_test)
    print('Train shape: ', x_train.shape, y_train.shape)
    print('Test shape: ', x_test.shape, y_test.shape)

    clf_svm = SVC(kernel='linear', probability=True) 
    clf_svm.fit(x_train, y_train)
    y_train_pred = clf_svm.predict(x_train)
    y_test_pred = clf_svm.predict(x_test)
    print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
    print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))
    return clf_svm

def classify_window(win, hog, clf_svm):
    win = cv2.resize(win,(140,60),interpolation=cv2.INTER_AREA)
    features = hog.compute(win)
    features = features.reshape(1, -1)
    return clf_svm.predict_proba(features)[0][1]

def process_image(image, hog, clf_svm):

    score = classify_window(image, hog, clf_svm)
    return score

# itest = load_image('vehicle_license_plate_test_dataset/test-vb-03.jpg')
# display_image(itest)
# itest.shape

# score, best_window, score_window = process_image(itest, step_size = 10, hog, clf_svm)
# print(score)
# print(score_window)
# display_image(best_window)  

def jaccard_index(true_box, predicted_box):
    y_a = max(true_box[0], predicted_box[0])
    x_a = max(true_box[1], predicted_box[1])
    y_b = min(true_box[2], predicted_box[2])
    x_b = min(true_box[3], predicted_box[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a +1)
    true_area = (true_box[3] - true_box[1] + 1) * (true_box[2] - true_box[0] + 1)
    pred_area = (predicted_box[3] - predicted_box[1] + 1) * (predicted_box[2] - predicted_box[0] + 1)
    retVal = inter_area/ float(true_area + pred_area - inter_area)
    return max(retVal, 0)
# print(jaccard_index([score_window[1], score_window[0], score_window[1] + 350, score_window[0]+ 400], [230,158,292,504]))