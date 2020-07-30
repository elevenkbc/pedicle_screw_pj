'''
Created on 2020年4月29日

@author: RaoBlack
'''

import cv2
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft


def getToolMap(img):
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
    return tool_map


file_name = "img0001.jpg" #vertical OK
# file_name = "img0002.jpg" #horizontal NG
# file_name = "img0003.jpg" #horizontal OK
# file_name = "img0004.jpg" #vertical OK

img = cv2.imread(os.path.join("dataset/img_no_ratation/", file_name))


im_h = img.shape[0]
im_w = img.shape[1]

center = [(im_h - 1) // 2, (im_h - 1) // 2]
radius = im_h // 2

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

img_gray = cv2.blur(img_gray, (5,5))


# new_img_gray = np.zeros((im_h, im_w), dtype = np.uint8)

# l_arr = [im_h//4-1, 2*im_h//4-1, 3*im_h//4-1]

l_arr = [im_h//2-1-20, im_h//2-1, im_h//2-1+20]



record = []
#抓縱線 三條垂直線
for l in l_arr:
    l_tmp = []
    for i in range(im_h):
        if math.sqrt((i - center[0]) ** 2 + (l - center[1]) ** 2) < (radius-2):
             
            l_tmp.append(img_gray[i, l])
            img[i, l, 2] = 255
            
    # fft
    fft_l_tmp = fft(l_tmp - np.mean(l_tmp, axis=0))
    abs_fft_l_tmp = np.abs(fft_l_tmp)
    
    record.append(np.mean(abs_fft_l_tmp))
    
    # fig, axs = plt.subplots(2)
    # fig.suptitle('vertical %d' %l)
    # axs[0].plot(l_tmp)
    # axs[1].plot(abs_fft_l_tmp)
    
    
    
#抓水平線 三條水平線
for l in l_arr:
    l_tmp = []
    for i in range(im_h):
        if math.sqrt((l - center[0]) ** 2 + (i - center[1]) ** 2) < (radius-2):
            
            l_tmp.append(img_gray[l, i])
            img[l, i, 2] = 255
            
    # fft
    
    fft_l_tmp = fft(l_tmp - np.mean(l_tmp, axis=0))
    abs_fft_l_tmp = np.abs(fft_l_tmp)
    
    record.append(np.mean(abs_fft_l_tmp))

    # fig, axs = plt.subplots(2)
    # fig.suptitle('horizontal %d' %l)
    # axs[0].plot(l_tmp)
    # axs[1].plot(abs_fft_l_tmp)
   
print(record)

# ret, img_gray2 = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY)

min_ind = record.index(min(record))

if min_ind < 4:
    print('vertical')
else:
    print('horizontal')

# cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img)
# cv2.imshow('new_img_gray', new_img_gray)
# cv2.imshow('img_gray2', img_gray2)

cv2.imwrite('tmp_output.jpg', img)

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
