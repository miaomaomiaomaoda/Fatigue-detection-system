import dlib
import imutils
import numpy as np
from imutils import face_utils
from cv2 import cv2
import matplotlib.pyplot as plt

# 第一步：使用dlib.get_frontal_face_detector() 获得脸部位置检测器
detector = dlib.get_frontal_face_detector()
# 第二步：使用dlib.shape_predictor获得脸部特征位置检测器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
cap = cv2.VideoCapture(0)

index = 0

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if index % 5 == 0:
        cv2.imwrite(str(index) + ".jpg", gray)
    index += 1
    cv2.imshow("Frame", frame)
    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

