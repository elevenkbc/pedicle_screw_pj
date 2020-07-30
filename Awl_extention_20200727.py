'''
Created on 2020年3月25日

@author: RaoBlack
'''
import cv2
from skimage import measure
import numpy as np
import scipy.io as scio
import os
import math

def remain_side_objects(img):
    labels = measure.label(img)
    h = img.shape[0]
    w = img.shape[1]
    left_side_flag = labels[0][0]
    right_side_flag = labels[0][h - 1]
    res_img = labels.copy()
    for i in range(h):
        for j in range(w):
            res_img_ij = res_img[i][j]
            if res_img_ij == left_side_flag or res_img_ij == right_side_flag:
                res_img[i, j] = 1
            else:
                res_img[i, j] = 0
    return res_img

    
# 處理 DEF

def remain_max_objects(img):
    labels = measure.label(img)  
    jj = measure.regionprops(labels)  
    # is_del = False
    if len(jj) == 1:
        del_mask = -1
    else:
        num = labels.max()  # 連通域的個數
        del_array = np.array([0] * (num + 1))
        for k in range(num):  # TODO：遇到圖片全黑，會出現錯誤
            if k == 0:
                initial_area = jj[0].area
                save_index = 1  # 初始保留第一個連通域
            else:
                k_area = jj[k].area  # 轉成array

                if initial_area < k_area:
                    initial_area = k_area
                    save_index = k + 1

        del_array[save_index] = 1
        del_mask = del_array[labels]
    return del_mask

def traspose_ct_img(img):
    im_h = img.shape[0]
    for i in range(im_h):
        for j in range(i, im_h):
            tmp_b = img[i][j][0]
            tmp_g = img[i][j][1]
            tmp_r = img[i][j][2]
            img[i][j][0] = img[j][i][0]
            img[i][j][1] = img[j][i][1]
            img[i][j][2] = img[j][i][2]
            img[j][i][0] = tmp_b
            img[j][i][1] = tmp_g
            img[j][i][2] = tmp_r
    return img

def matrix_save_mat(matrix, filename):
    # debug 用
    scio.savemat(filename + '.mat', {'A':matrix})



# imagePath = "/home/raoblack/Documents/darknet/Projects/Pedicle0617/Images/img0010.jpg"



# file_name = "img0008.jpg" #NG
# file_name = "img0009.jpg" #OK, for th = 165
# file_name = "img0010.jpg" #OK, for th = 147
# file_name = "img0011.jpg" #OK,  for th = 154
# file_name = "img0012.jpg" #OK,  for th = 154
# file_name = "img0013.jpg" #OK,  for th = 144
# file_name = "img0014.jpg" #OK,  for th = 144
# file_name = "img0015.jpg" #OK,  for th = 144
# file_name = "img0016.jpg" #OK,  for th = 144
# file_name = "img0017.jpg" #OK,  for th = 144
# file_name = "img0018.jpg" #OK,  for th = 144
# file_name = "img0019.jpg" #OK,  for th = 144
# file_name = "img0020.jpg" #OK,  for th = 144
# file_name = "img0023.jpg" #OK,  for th = 144
# file_name = "img0024.jpg" #OK,  for th = 144
# file_name = "img0025.jpg" #NG,  for th = 144
# file_name = "img0026.jpg" #NG,  for th = 144
# file_name = "img0027.jpg" #NG,  for th = 144
# file_name = "img0028.jpg" #NG,  for th = 144
# file_name = "img0030.jpg" #OK,  for th = 138
# file_name = "img0031.jpg" #OK,  for th = 138
# file_name = "img0032.jpg" #OK,  for th = 138
# file_name = "img0033.jpg" #NG,  for th = 138
# file_name = "img0034.jpg" #NG,  for th = 138
file_name = "img0035.jpg" #NG,  for th = 138

img = cv2.imread(os.path.join("dataset/awl/", file_name))
img = traspose_ct_img(img)
img_org = img.copy()



# cv2.imshow('origin_img', img_org)
height = img.shape[0]  # 高度
width = img.shape[1]  # 寬度
print('width = {}, height = {}'.format(width, height))

# 負片效果:
b, g, r = cv2.split(img)
b = 255 - b
g = 255 - g
r = 255 - r
# change the arrays's value by indexing
img[:, :, 0] = b
img[:, :, 1] = g
img[:, :, 2] = r
# cv2.imshow('reverse img', img)

# 顏色反轉
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)


th = 148

ret, thresh1 = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
# cv2.imshow('thresh1', thresh1)
# cv2.waitKey(0)


thresh2 = thresh1.copy()

r = height // 2 - 5
ceter = [height // 2, height // 2]
for i in range(width): 
    for j in range(height):
        if ((i - ceter[0]) ** 2 + (j - ceter[1]) ** 2 > r ** 2):
            thresh2[j][i] = 0

for i in range(width):
    for j in range(height):
        if i > 9.5 * height / 10 and i < height:
            thresh2[j][i] = 255
        if i < 0.5 * height / 10:
            thresh2[j][i] = 255 
        if (i - ceter[0]) ** 2 + (j - ceter[1]) ** 2 < r ** 2 and j > 8 * height / 10:
            thresh2[j][i] = 0
        if (i - ceter[0]) ** 2 + (j - ceter[1]) ** 2 < r ** 2 and j < 2 * height / 10:
            thresh2[j][i] = 0
            
# cv2.imshow('thresh2', thresh2)
thresh3 = (remain_side_objects(thresh2) * 255).astype(np.uint8)  # 留下兩側


# cv2.imshow('thresh3', thresh3)
thresh4 = (remain_max_objects(thresh3) * 255).astype(np.uint8)  # 留下最大的


for i in range(width): 
    for j in range(height):
        if ((i - ceter[0]) ** 2 + (j - ceter[1]) ** 2 > r ** 2):
            thresh4[j][i] = 0
            
# cv2.imshow('thresh4', thresh4)

thresh5 = np.bitwise_and(thresh1, thresh4)
# cv2.imshow('thresh5', thresh5)

# median filter
thresh6 = cv2.medianBlur(thresh5, 9)
# cv2.imshow('thresh6', thresh6)

edge_contour = cv2.Canny(thresh6, 50, 150, apertureSize=3)  # edge detector
# cv2.imshow('edge_contour', edge_contour)

midline = np.zeros((height, width), dtype=np.uint8)
for i in range(width):
    col = edge_contour[:, i]
    ind_arr = np.where(col == 255)[0]
    if len(ind_arr) != 0:
        if (max(ind_arr) - min(ind_arr)) < height * 60 // 1024:
            md = (max(ind_arr) + min(ind_arr)) / 2
            midline[int(round(md)), i] = 255
               
# cv2.imshow('midline', midline)

# line_select = np.zeros((height, width), dtype=np.uint8)
hough_result = img.copy()
            
# lines = cv2.HoughLines(midline, 1, np.pi / 180, 85)
# lines = cv2.HoughLines(midline, 1, np.pi / 180, 70)
lines = cv2.HoughLines(midline, 1, np.pi / 180, 40)
if lines is None:
    raise Exception("Not find awl in the photo")
lines = lines[:, 0, :]  # 3D提取為2D
lines1 = lines[0]  # 只拿第一條線

hough_lines_mask = np.zeros((height, width), dtype=np.uint8)

rho = lines1[0]
theta = lines1[1]
# x0 = rho*cos(theta): x-axis intercept
# y0 = rho*sin(theta): y-axis intercept
# slope: - a/b
# equation:  (y - y0) = (-a/b)*(x - x0)
# => y = - (a/b)*x + ((a/b)*x0 + y0)
a = np.cos(theta)
b = np.sin(theta)
x0 = a * rho
y0 = b * rho
p_left = [0, int(round((a / b) * x0 + y0))]
p_right = [width - 1, int(round(-(a / b) * (width - 1) + ((a / b) * x0 + y0)))]

# p_left = [int(x0 + 0*(-b)), int(y0 + 0*(a))]
# p_right = [int(x0 - 1000*(-b)), int(y0 - 1000*(a))]

cv2.line(hough_lines_mask, (p_left[0], p_left[1]), (p_right[0], p_right[1]), 255, 1)

# cv2.imshow('hough_lines_mask', hough_lines_mask)

hough_move_test = np.zeros((height, width), dtype=np.uint8)

# x0 = rho*cos(theta): x-axis intercept
# y0 = rho*sin(theta): y-axis intercept
# slope: - a/b
# equation:  y = y0 - (a / b) * (x - x0)
a = np.cos(theta)
b = np.sin(theta)
x0 = a * rho
y0 = b * rho

# 中線的斜率為 - a/b
dx = b / (np.sqrt(a ** 2 + b ** 2))
dy = -a / (np.sqrt(a ** 2 + b ** 2))

# 垂直的斜率為 b/a ， 並且 normalized
dx_p = a / (np.sqrt(a ** 2 + b ** 2))
dy_p = b / (np.sqrt(a ** 2 + b ** 2))
# 對線上每點 vertical search
pts = []
pre = False

# modify
detect_map = thresh6.copy()
pointed_map = np.zeros((height, width), dtype=np.uint8)
t = 0
while True:
    xi = p_left[0] + dx * t
    yi = p_left[1] + dy * t
    if 0 > xi or xi > width - 1  or 0 > yi or yi > height - 1:
        break
    
    # 在中線的垂直方向找
    up_p = [-1]
    for ii in range(100, 0, -1):
        xii = int(round(xi + dx_p * ii))
        yii = int(round(yi + dy_p * ii))
        if (0 < xii and xii < width - 1  and 0 < yii and yii < height - 1 
            and (detect_map[yii][xii] or detect_map[yii + 1][xii + 1] 
            or detect_map[yii + 1][xii] or detect_map[yii + 1][xii - 1] 
            or detect_map[yii][xii + 1] or detect_map[yii][xii - 1] 
            or detect_map[yii - 1][xii + 1] or detect_map[yii - 1][xii] 
            or detect_map[yii - 1][xii - 1])):
            up_p = [int(round(xii)), int(round(yii))]
            break
    
    lower_p = [-1]
    for ii in range(-100, 0 , 1):
        xii = int(round(xi + dx_p * ii))
        yii = int(round(yi + dy_p * ii))
        if (0 < xii and xii < width - 1  and 0 < yii and yii < height - 1 
            and (detect_map[yii][xii] or detect_map[yii + 1][xii + 1] 
            or detect_map[yii + 1][xii] or detect_map[yii + 1][xii - 1] 
            or detect_map[yii][xii + 1] or detect_map[yii][xii - 1] 
            or detect_map[yii - 1][xii + 1]or detect_map[yii - 1][xii] 
            or detect_map[yii - 1][xii - 1])):    
            lower_p = [int(round(xii)), int(round(yii))]
            break
    
#     Pointed_len = height * 30 // 1024
    Pointed_len = height * 33 // 1024
    #     Pointed_len = 15 
    #     Pointed_len = 43 
    flag = len(up_p) == 2 and len(lower_p) == 2 and np.linalg.norm([up_p[0] - lower_p[0], up_p[1] - lower_p[1]]) < Pointed_len
        
    if flag ^ pre:
        pts.append([int(round(xi)), int(round(yi))])
        if len(pts) == 1 and len(up_p) == 2 and len(lower_p) == 2:
            cv2.line(pointed_map, (up_p[0], up_p[1]), (lower_p[0], lower_p[1]), 255, 1)
        if len(pts) == 2 and len(up_p) == 2 and len(lower_p) == 2:
            cv2.line(pointed_map, (up_p[0], up_p[1]), (lower_p[0], lower_p[1]), 255, 1)
            break
     
    if flag:
        pointed_map[up_p[1], up_p[0]] = 255
        pointed_map[lower_p[1], lower_p[0]] = 255
        pre = flag
    else:
        pre = False
    t += 1

p1 = pts[0]
p2 = pts[1]

Result = img_org.copy()

# 確定方向
dir = 0
for i in range(height // 10):
    if max(thresh6[:, i]) > 0:
        dir = 1  # 方向是左邊
        break
for i in range(height, 9 * height // 10, -1):
    if max(thresh6[:, i]) > 0:
        dir = -1  # 方向是右邊
        break
# 將 pointed_map 延伸
if dir == 1:
    ext_dir = [p2[0] - p1[0], p2[1] - p1[1]]
elif dir == -1:
    ext_dir = [p1[0] - p2[0], p1[1] - p2[1]]
else:
    raise Exception("dir not defined")
t = 0
# draw_color = (0, 255, 255) #yellow
draw_color = (0, 0, 255)  # red
for i in range(width):
    for j in range(height):
        shift_i = i + ext_dir[0]
        shift_j = j + ext_dir[1]
        if pointed_map[j][i] == 255 and (0 < shift_i and shift_i < width - 1  and 0 < shift_j and shift_j < height - 1):
            Result[shift_j][shift_i] = draw_color
        if ((i - ceter[0]) ** 2 + (j - ceter[1]) ** 2 < r ** 2) and (hough_lines_mask[j][i] == 255):
            Result[j][i] = draw_color
# while True:
#     xi = (1 - t) * p1[0] + t * p2[0]
#     yi = (1 - t) * p1[1] + t * p2[1]
#     if t > 1:
#         break
# 
#     for ii in range(-49, 50):
# #         print('ii={}'.format(ii))
#         xii = int(round(xi + dx_p * ii))
#         yii = int(round(yi + dy_p * ii))
# #         hough_move_test[yii][xii] = 1
#         if (0 < xii and xii < width - 1  and 0 < yii and yii < height - 1 
#             and (edge_contour[yii][xii] or edge_contour[yii + 1][xii + 1] 
#             or edge_contour[yii + 1][xii] or edge_contour[yii + 1][xii - 1] 
#             or edge_contour[yii][xii + 1] or edge_contour[yii][xii - 1] 
#             or edge_contour[yii - 1][xii + 1]or edge_contour[yii - 1][xii] or edge_contour[yii - 1][xii - 1])):
#             hough_move_test[yii + ext_dir[1]][xii + ext_dir[0]] = 1
#             Result[yii + ext_dir[1]][xii + ext_dir[0]] = (255, 50, 50)
#     t += 0.005
# cv2.imshow('hough_move_test', 255 * hough_move_test.astype(np.uint8))

Result = traspose_ct_img(Result)

rdeg = math.atan(dy/dx)
ddeg = rdeg*(180/math.pi)
text = 'Angle of awl = %.2f (degree)'%ddeg
cv2.putText(Result, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
  1, (0, 0, 255), 2, cv2.LINE_AA)

SaveFolder = 'My_Result20200730'

if not os.path.exists(os.path.join(SaveFolder, 'awl_extention')):
    os.makedirs(os.path.join(SaveFolder, 'awl_extention'))
cv2.imshow('Result', Result)
cv2.imwrite(os.path.join(SaveFolder, 'awl_extention', file_name), Result)

cv2.waitKey(0)
cv2.destroyAllWindows()
