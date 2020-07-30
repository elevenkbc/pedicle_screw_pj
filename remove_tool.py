'''
Created on 2020年7月30日

@author: RaoBlack
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# imagePath = "/home/raoblack/Documents/darknet/Projects/Pedicle0617/Images/img0006.jpg"

# file_name = "img0008.jpg" #NG
# file_name = "img0024.jpg" #OK
# file_name = "img0026.jpg" #OK
# file_name = "img0030.jpg" #OK
file_name = "img0031.jpg" #OK

img = cv2.imread(os.path.join("dataset/img/", file_name))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #轉成灰階值


histSize = 32
dh = 256//histSize #interval of histogram bins

xray_mask = np.zeros(img_gray.shape, np.uint8)
center = [img_gray.shape[0]//2, img_gray.shape[0]//2]
for i in range(xray_mask.shape[0]):
    for j in range(xray_mask.shape[1]):
        if (int(np.sqrt((i - center[0])**2 + (j - center[1])**2)) <= center[0]):
            xray_mask[i, j] = 255


# cv2.imshow("xray_mask", xray_mask)
# cv2.waitKey(0)
# print("xray_mask.shape = ", xray_mask.shape)
# print("img_gray.shape = ", img_gray.shape)

hist = cv2.calcHist([img_gray], [0], xray_mask, [histSize], [0, 256])
# hist = np.array([[1], [7], [5], [9], [2], [1], [4]])

cv2.imshow("img", img)

tool_map = np.zeros(img_gray.shape, dtype = np.uint8)

r = 0.15
hist_sum = np.sum(hist, axis = 0)[0]

# print('hist_sum = ', hist_sum)
left_tail_num = hist_sum*r

left_tail_ind = 0
tmp_sum = 0
for i in range(hist.shape[0]):
    tmp_sum += hist[i, 0]
    if tmp_sum >= left_tail_num:
        left_tail_ind = i
        break


for i in range(img_gray.shape[0]):
    for j in range(img_gray.shape[1]):
        if xray_mask[i, j] == 255:
            if img_gray[i, j] < dh*(left_tail_ind  + 1):
                tool_map[i, j] = 255
# cv2.imshow("tool_map", tool_map)


#fill zero value at tool_map
for i in range(img_gray.shape[0]):
    for j in range(img_gray.shape[1]):
        if tool_map[i, j] == 255:
            img_gray[i, j] = 255
cv2.imshow("img_gray", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


SaveFolder = 'My_Result20200730'

if not os.path.exists(os.path.join(SaveFolder, 'remove_tools')):
    os.makedirs(os.path.join(SaveFolder, 'remove_tools'))

cv2.imwrite(os.path.join(SaveFolder, 'remove_tools', file_name), img_gray)