import cv2.cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('310.jpg', 0)
clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
gray = clahe.apply(img)
# 显示图像
res = np.hstack((img, gray))
cv2.imshow("frame", res)
cv2.waitKey()
cv2.imwrite('res.jpg', res)
