# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 16:42:10 2021

@author: Leon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from skimage.io import imread, imshow
from sklearn.mixture import GaussianMixture
import cv2

#//切掉透明
def delTransparent(array, yn):
    image = []
    ##array[圖][列][像素][通道]
    for i in range(len(array)):
        if array[i][3] != 0:
            if(yn == 1):
                array[i][3] = 255
            else:
                array[i][3] = 0
            image.append(array[i])
    return image

#//讀取圖片
image = cv2.imread('full_duck.jpg')
image = np.reshape(image, (-1, 3))
print(np.shape(image))

#//show圖片
#cv2.imshow('duck', image)
#cv2.waitKey(0)

#//鴨子Yes的data處理
img_y = []
train_y = []
for i in range(1, 4):
    img_y.append(cv2.imread('duck_y' + str(i) + '.png', cv2.IMREAD_UNCHANGED))
    
for i in range(0, 3):
    img_y[i] = np.reshape(img_y[i], (-1, 4))
    train_y.append(delTransparent(img_y[i], 1))
    
train_y = np.array(train_y)

#//鴨子No的data處理
img_n = []
train_n = []
for i in range(1, 4):
    img_n.append(cv2.imread('duck_n' + str(i) + '.png', cv2.IMREAD_UNCHANGED))
    
for i in range(0, 3):
    img_n[i] = np.reshape(img_n[i], (-1, 4))
    train_n.append(delTransparent(img_n[i], 0))

train_n = np.array(train_n)

#//合成鴨子Yes & No
all_data = []
for i in range(0, 3):
    all_temp = np.append(train_y[i], train_n[i], axis= 0)
    all_data.append(all_temp)
temp = np.append(all_data[0], all_data[1], axis = 0)
final_all = np.append(all_data[2], temp, axis = 0)
print(final_all)
print(np.shape(final_all))
import random
random.seed(2)
random.shuffle(final_all)

#//測試殘渣
#print(np.shape(img_y[0]))
#print(len(train_y[0]))
#while len(train_y) != 0:
#print(train_y)
#gm_yes = GaussianMixture(n_components=3, random_state=0).fit(train_y)
#print(gm_yes.predict([0,0,0],[255,255,255]))

#//Naive Bayes modoel
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(final_all[:, :3], final_all[:, 3], test_size=0.1)
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

#//可視化
image_pred = classifier.predict(image)
#//轉回來原本shape = (13816, 5946, 3)
image_pred = np.reshape(image_pred, (13816, 5946))

#//去雜訊
kernel = np.ones((3,3), np.uint8)
image_pred = cv2.dilate(image_pred, kernel, iterations = 1)
image_pred = cv2.erode(image_pred, kernel, iterations = 3)
image_pred = cv2.dilate(image_pred, kernel, iterations = 2)

#//寫檔案
cv2.imwrite('duck_predict.jpg', image_pred)
