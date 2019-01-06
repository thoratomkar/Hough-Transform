# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:14:28 2018

@author: omkar
"""
import cv2
import numpy as np

def image_display(image,name):
    cv2.imshow(name, image)    
    cv2.imwrite(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def erode(img):
    
    rows = img.shape[0]
    columns = img.shape[1]
    blank = np.zeros((rows,columns))
    
    for i in range (0,rows):
        for j in range (0, columns):
            row1=False
            row2=False
            row3=False
            if img[i][j] == 255 or img[i][j] == 0:
                if (i>0) and (j>0) and (i+1<rows) and (j+1<columns):
                    #print(i,j)
                    if (img[i-1][j-1] == 255 and img[i-1][j] == 255 and img[i-1][j+1] == 255):
                        row1=True
                    if (img[i][j-1] == 255 and img[i][j+1] == 255):
                        row2=True
                    if (img[i+1][j-1] == 255 and img[i+1][j] == 255 and img[i+1][j+1] == 255):
                        row3=True
                    if row1==True and row2==True and row3==True:
                        #print('here')
                        blank[i][j] = 255
                    else:
                        blank[i][j] = 0
            
    return blank.astype(np.uint8)


def dilate(img):
    
    rows = img.shape[0]
    columns = img.shape[1]
    blank = np.zeros((rows,columns))
    
    for i in range (0,rows):
        for j in range (0, columns):
            row1=False
            row2=False
            row3=False
            if img[i][j] == 255 :
                if (i>0) and (j>0) and (i+1<rows) and (j+1<columns):
                    #print(i,j)
                    if (img[i-1][j-1] == 255 or img[i-1][j] == 255 or img[i-1][j+1] == 255):
                        row1=True
                    if (img[i][j-1] == 255 or img[i][j] or img[i][j+1] == 255):
                        row2=True
                    if (img[i+1][j-1] == 255 or img[i+1][j] == 255 or img[i+1][j+1] == 255):
                        row3=True
                    if row1==True or row2==True or row3==True:
                        #print('here')
                        blank[i-1][j-1] = 255
                        blank[i-1][j] = 255
                        blank[i-1][j+1] = 255
                        blank[i][j-1] = 255
                        blank[i][j] = 255
                        blank[i][j+1] = 255
                        blank[i+1][j-1] = 255
                        blank[i+1][j] = 255
                        blank[i+1][j+1] = 255
                    else:
                        blank[i][j] = 0
            
    return blank.astype(np.uint8)


def opening(img):
    e_img = erode(img)
    d_img = dilate(e_img)
    return d_img


def closing(img):
    d_img = dilate(img)
    e_img = erode(d_img)
    return e_img


image = cv2.imread('task1.png',0)

#Algorithm 1 - Opening
res_img1 = opening(image)
image_display(res_img1,'res_noise1.jpg')
#extracting boundary
e_image = erode(res_img1)
bound1 = cv2.subtract(res_img1,e_image)
image_display(bound1,'res_bound1.jpg')

#Algorithm 2 - Closing
res_img2 = closing(image)
image_display(res_img2,'res_noise2.jpg')
#extracting boundary
e_image = erode(res_img2)
bound2 = cv2.subtract(res_img2,e_image)
image_display(bound2,'res_bound2.jpg')