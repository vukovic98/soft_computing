import imutils as imutils
import inline as inline
import matplotlib
import numpy as np
import cv2
import pytesseract as pytesseract
from sklearn import datasets
import matplotlib.pyplot as plt

def process_video(video_path):

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

        img = frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale

        gray = cv2.bilateralFilter(gray, 13, 15, 15)

        edged = cv2.Canny(gray, 30, 200)  # Perform Edge detection
        plt.imshow(edged)
        plt.show()
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        screenCnt = None

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
                print(rectness, approx)
                possibleContoures.append((rectness, approx))
                # screenCnt = approx
                # break

        val = max(possibleContoures,key=lambda item:item[0])[1]


        plt.imshow(img)
        plt.show()

        # Masking the part other than the number plate
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [val], 0, 255, -1, )
        new_image = cv2.bitwise_and(img, img, mask=mask)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        cropped_image = gray[topx:bottomx + 1, topy:bottomy + 1]
        
        plt.imshow(new_image)
        plt.show()
        break

    cap.release()
    return True

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
    process_video("./data/video_Trim.mp4")