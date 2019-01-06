# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 16:21:09 2018

@author: omkar
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def boundingboxes(img, processed_image):
    
    (_, contours, _) = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    l = []
    # print table of contours and sizes
    print("Found %d objects." % len(contours))
    
    for contour in contours:
        
        x,y,w,h = (cv2.boundingRect(contour))
        
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        l.append(box)
        # draw a red 'nghien' rectangle
        cv2.drawContours(img, [box], -1, (255, 0, 0),5)
    
    l=np.array(l)
    image_display(img,'bounding_box_2a_color.jpg')

def image_display(image,name):
    
    cv2.imshow(name, image)    
    cv2.imwrite(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def point_detection(img, mask):
    
    t = 650
    #t = 2250
    rows = img.shape[0]
    columns = img.shape[1]
    img_point = []
    img_point = np.zeros((len(img),len(img[0])))
        
    for i in range(0,rows):
        for j in range(0,columns):
            if i> 0 and i< rows-1 and j>0 and j<columns-1:
                
                r=(mask[0][0]*img[i-1][j-1])+(mask[0][1]*img[i-1][j])+(mask[0][2]*img[i-1][j+1])+\
                   (mask[1][0]*img[i][j-1])+(mask[1][1]*img[i][j])+(mask[1][2]*img[i][j+1])+\
                   (mask[2][0]*img[i+1][j-1])+(mask[2][1]*img[i+1][j])+(mask[2][2]*img[i+1][j+1])
                
                '''
                r=(mask[0][0]*img[i-2][j-2])+(mask[0][1]*img[i-2][j-1])+(mask[0][2]*img[i-2][j])+(mask[0][3]*img[i-2][j+1])+(mask[0][4]*img[i-2][j+2])+\
                   (mask[1][0]*img[i-1][j-2])+(mask[1][1]*img[i-1][j-1])+(mask[1][2]*img[i-1][j])+(mask[1][3]*img[i-1][j+1])+(mask[1][4]*img[i-1][j+2])+\
                   (mask[2][0]*img[i][j-2])+(mask[2][1]*img[i][j-1])+(mask[2][2]*img[i][j])+(mask[2][3]*img[i][j+1])+(mask[2][4]*img[i][j+2])+\
                   (mask[3][0]*img[i+1][j-2])+(mask[3][1]*img[i+1][j-1])+(mask[3][2]*img[i+1][j])+(mask[3][3]*img[i+1][j+1])+(mask[3][4]*img[i+1][j+2])+\
                   (mask[4][0]*img[i+2][j-2])+(mask[4][1]*img[i+2][j-1])+(mask[4][2]*img[i+2][j])+(mask[4][3]*img[i+2][j+1])+(mask[4][4]*img[i+2][j+2])
                '''
                
                if abs(r)>=t:   
                    img_point[i][j]=255
                    print((i,j))
    
    return img_point.astype(np.uint8)

def histogram(img):
    d={}
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if img[i][j] in d:
                d[img[i][j]] +=1
            else:
                d[img[i][j]] = 1
    
    
    k = list(d.keys())
    v = list(d.values())
    
    diff =[]
    indices =[]
    
    # taking group of 5 pixels to check the thresholding
    for i in range(6,len(v)-5):
        x1 = v[i-5]
        x2 = v[i+5]
        y =abs(x1-x2)
        diff. append(y)
        indices.append(i+5)
    y = max(diff)
    idx = diff.index(y)
    
    plt.bar(k, v, color='g')
    plt.axis([0,255,0,1850])
    plt.show()
    
    return indices[idx]


def segmentation(img,t):
    
    rows = img.shape[0]
    columns = img.shape[1]
    
    img_point = np.zeros((len(img),len(img[0])))
    
    for i in range(0,rows):
        for j in range(0,columns):
            
            if img[i][j] >= t:   
                img_point[i][j]=255
            else:
                img_point[i][j]=0
            
    return img_point.astype(np.uint8)

def draw_lines(img, a, b, c, d):
    
    cv2.line(img, a, b, (0,0,255), 1) 
    cv2.line(img, b, c, (0,0,255), 1) 
    cv2.line(img, c, d, (0,0,255), 1) 
    cv2.line(img, d, a, (0,0,255), 1) 
    

img = cv2.imread('turbine-blade.jpg')
bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#mask = np.array([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, 24, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]], dtype = np.float)
mask = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype = np.float)
processed_image = point_detection(bw, mask)
image_display(processed_image,'Task 2a.jpg')
boundingboxes(img,processed_image)

img = cv2.imread('segment.jpg')
bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

t = histogram(bw)
print('The threshold is: ',t)
print('\nSegmenting the image on the threshold obtained.')
processed_image = segmentation(bw, t)
image_display(processed_image,'Task 2b.jpg')

draw_lines(img, (163,126), (201,126), (201,165), (163,165))
draw_lines(img, (252,78), (305,78), (305,204), (252,204))
draw_lines(img, (336,22), (362,22), (362,288), (336,288))
draw_lines(img, (388,42), (428,42), (428,255), (388,255))
image_display(img, 'Task 2b Boxes_color.jpg')