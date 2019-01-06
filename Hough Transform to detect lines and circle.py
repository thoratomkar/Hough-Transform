import cv2
import numpy as np
import operator
from copy import deepcopy

def sobel_filter(img, sobel_diagonal, sobel_vertical):
    
    rows = img.shape[0]
    columns = img.shape[1]
    img_hori = []
    img_vertical = []
        
    for i in range(rows):
        rowList = []
        for j in range(columns):            
            rowList.append(0)
        img_hori.append(rowList)
        
    for i in range(rows):
        rowList = []
        for j in range(columns):            
            rowList.append(0)
        img_vertical.append(rowList)
        
    for i in range(0,rows):
        for j in range(0,columns):
            if i>1 and j>1 and i<rows-2 and j<columns-2:
                gx=(sobel_diagonal[0][0]*img[i-1][j-1])+(sobel_diagonal[0][1]*img[i-1][j])+(sobel_diagonal[0][2]*img[i-1][j+1])+\
                   (sobel_diagonal[1][0]*img[i][j-1])+(sobel_diagonal[1][1]*img[i][j])+(sobel_diagonal[1][2]*img[i][j+1])+\
                   (sobel_diagonal[2][0]*img[i+1][j-1])+(sobel_diagonal[2][1]*img[i+1][j])+(sobel_diagonal[2][2]*img[i+1][j+1])
                
                gy=(sobel_vertical[0][0]*img[i-1][j-1])+(sobel_vertical[0][1]*img[i-1][j])+(sobel_vertical[0][2]*img[i-1][j+1])+\
                   (sobel_vertical[1][0]*img[i][j-1])+(sobel_vertical[1][1]*img[i][j])+(sobel_vertical[1][2]*img[i][j+1])+\
                   (sobel_vertical[2][0]*img[i+1][j-1])+(sobel_vertical[2][1]*img[i+1][j])+(sobel_vertical[2][2]*img[i+1][j+1])
                
                if gx>30 or gy>30:
                    img_hori[i-1][j-1]=gx
                    img_vertical[i-1][j-1]=gy
    return np.array(img_hori, dtype=np.uint8), np.array(img_vertical, dtype=np.uint8)  


def HoughCircles(input,circles): 
    rows = input.shape[0] 
    cols = input.shape[1] 
    
    # initializing the angles to be computed 
    sinang = dict() 
    cosang = dict() 
    Threshold = 150
    # initializing the angles  
    for angle in range(0,360): 
        sinang[angle] = np.sin(angle * np.pi/180) 
        cosang[angle] = np.cos(angle * np.pi/180) 
            
    # initializing the different radius
    
    radius = [i for i in range(10,25)]
        
    for r in radius:
        #Initializing an empty 2D array with zeroes 
        acc_cells = np.zeros([rows,cols]) 
        # Iterating through the original image 
        for x in range(rows): 
            for y in range(cols): 
                #checking edge
                if input[x][y] == 255: 
                    # increment in the accumulator cells 
                    for angle in range(0,360): 
                        b = y - round(r * sinang[angle]) 
                        a = x - round(r * cosang[angle]) 
                        if a >= 0 and a < rows and b >= 0 and b < cols: 
                            acc_cells[int(a)][int(b)] += 1
                             
        print('For radius: ',r)
        acc_cell_max = np.amax(acc_cells)
        print('max acc value: ',acc_cell_max)
        
        if(acc_cell_max > Threshold):  

            print("Detecting the circles for radius: ",r)       
            
            # Initial threshold
            acc_cells[acc_cells < Threshold] = 0  
               
            # find the circles for this radius 
            for i in range(rows): 
                for j in range(cols): 
                    if(i > 0 and j > 0 and i < rows-1 and j < cols-1 and acc_cells[i][j] >= 150):
                        avg_sum = np.float32((acc_cells[i][j]+acc_cells[i-1][j]+acc_cells[i+1][j]+acc_cells[i][j-1]+acc_cells[i][j+1]+acc_cells[i-1][j-1]+acc_cells[i-1][j+1]+acc_cells[i+1][j-1]+acc_cells[i+1][j+1])/9) 
                        if(avg_sum >= 33):
                            circles.append((i,j,r))
                            acc_cells[i:i+5,j:j+7] = 0


def gaussian_smoothing(input_img):
                                
    gaussian_filter=np.array([[0.109,0.111,0.109],
                              [0.111,0.135,0.111],
                              [0.109,0.111,0.109]])
                                
    return cv2.filter2D(input_img,-1,gaussian_filter)  
        
def canny_edge_detection(input):
    
    input = input.astype('uint8')

    # Using OTSU thresholding
    otsu_threshold_val, ret_matrix = cv2.threshold(input,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
    lower_threshold = otsu_threshold_val * 0.4
    upper_threshold = otsu_threshold_val * 1.3
    
    print(lower_threshold,upper_threshold)
    
    edges = cv2.Canny(input, lower_threshold, upper_threshold)
    return edges


def hough_lines(img, t1, t2, step_size, init_theta = 1, init_rho = 1):
   
    rows = img.shape[0]
    columns = img.shape[1]
    
    Diagonal = np.sqrt((rows)**2 + (columns)**2)
    
    Q = np.ceil(Diagonal/init_rho)
    rho = 2*Q + 1
    rho_grid = np.linspace(-Q*init_rho, Q*init_rho, rho)
    #theta_grid = np.linspace(0,180,np.ceil(180/init_theta)+1)
    theta_grid = np.linspace(t1, t2, np.ceil(30.0/init_theta) + step_size)
    theta_grid = np.concatenate((theta_grid, -theta_grid[len(theta_grid)-2::-1]))
    
    H = np.zeros([len(rho_grid),len(theta_grid)])
    
    for i in range (rows):
        for j in range (columns):
            if img[i][j]:
                for t in range(len(theta_grid)):
                    rho_val = j*np.cos(theta_grid[t]*np.pi/180.0) + i*np.sin(theta_grid[t]*np.pi/180.0)
                    for r_index in range(0, len(rho_grid)):
                        if rho_grid[r_index]>rho_val:
                            break
                    H[r_index][t] += 1                      
    
    return rho_grid, theta_grid, H


def get_rho_theta_pairs(x_y_pairs, rho, theta):
    
    rho_theta_pair=[]
    for i in range(0,len(x_y_pairs)):
        result=x_y_pairs[i]
        
        rhos = rho[result[1]]
        thetas = theta[[result[0]]]
        rho_theta_pair.append([rhos,thetas[0]])
    return rho_theta_pair


def voting(H,rho,theta, n_lines):
   
    d={}
    x_y_pairs = []    
    k = []
    
    for i in range(0,len(rho)):
        for j in range(0,len(theta)):
           key=(i,j)
           
           if(key in d):
               d[key]+=H[i][j]
           else:
               d[key]=H[i][j]
    
    k = d.items()
    
    for i in range(0,n_lines):
        key=max(k, key=operator.itemgetter(1))[0]
        r_key=key[::-1]
        x_y_pairs.append(r_key)
        del d[key]
    
    rho_theta_pair = get_rho_theta_pairs(x_y_pairs, rho, theta)    
           
    return x_y_pairs,rho_theta_pair


def plot_hough_lines(img, rho_theta_pairs, color, name):
      
    max_y = img.shape[0]
    max_x = img.shape[1]
    
    for i in range(0, len(rho_theta_pairs), 1):
        
      point = rho_theta_pairs[i]
      rho = int(point[0])
      theta = (point[1]) * np.pi / 180 # degrees to radians
           
      # y = mx + b form
      m = -np.cos(theta) / np.sin(theta)
      b = rho / np.sin(theta)
      # possible intersections on image edges
      left = (0, b)
      right = (max_x, max_x * m + b)
      top = (-b / m, 0)
      bottom = ((max_y - b) / m, max_y)
      
      pts = []
      for point in [left, right, top, bottom]:
          
          x, y = point
          if x <= max_x and x >= 0 and y <= max_y and y >= 0:
              pts.append(point)
             
      if len(pts) == 2:
          
          l = []
          for i in range(0, len(pts)):
              p = pts[i]
              x = int(round(p[0]))
              y = int(round(p[1]))
              l.append((x,y))
          cv2.line(img, l[0], l[1], color, 2)            
        
    cv2.imshow(name,img)
    cv2.imwrite(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img_orig = cv2.imread('hough.jpg')
img = img_orig[:,:,::-1]
bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Edge Detection
print("Performing Edge Detection")
sobel_diagonal = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype = np.float)
sobel_vertical = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype = np.float)
#calling sobel to perform edge detection
img_diag, img_vertical = sobel_filter(bw, sobel_diagonal, sobel_vertical)
cv2.imwrite('Diagonal Edges.jpg',img_diag)
cv2.imwrite('Vertical Edges.jpg',img_vertical)
print("Edge Detection Done.\n\nDetecting Vertical Lines\n\n")

#Detect Vertical Lines
rhos, thetas, H = hough_lines(img_vertical,0,35,1.0)
x_y_pairs, rho_theta_pairs = voting(H,rhos,thetas,22)
final_image = img.copy()
plot_hough_lines(final_image, rho_theta_pairs,(0,0,255),'red_line.jpg')
print("Vertical Lines Detected.\n\nDetecting Diagonal Lines")

#Detect Diagonal Values
rhos, thetas, H = hough_lines(img_diag,-60,-30,0.0)
x_y_pairs, rho_theta_pairs = voting(H,rhos,thetas,22)
final_image = img.copy()
plot_hough_lines(final_image, rho_theta_pairs,(255,0,0),'blue_lines.jpg')

#Detect Coins

circles = []
input = cv2.imread('hough.jpg',cv2.IMREAD_GRAYSCALE)
input_img = deepcopy(input)
smoothed_img = gaussian_smoothing(input_img)
    
   
edged_image = canny_edge_detection(smoothed_img)

HoughCircles(edged_image,circles) 

# Print the output
for vertex in circles:
    cv2.circle(img_orig,(vertex[1],vertex[0]),vertex[2],(0,255,0),1)
    cv2.rectangle(img_orig,(vertex[1]-2,vertex[0]-2),(vertex[1]-2,vertex[0]-2),(0,0,255),3)
    
   
cv2.imshow('Circle Detected Image',img_orig) 
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('Circle_Detected_Image.jpg',img_orig) 