from turtle import width
from matplotlib import lines
import matplotlib.pylab as plt
import cv2
from matplotlib.pyplot import gray
import numpy as np

################################################################################################################

image = cv2.imread('/home/uduwela/co4204_lab_2_road_lane_detection-dhanushka365/TestVideo_1/Right_0.bmp')
lane_image = np.copy(image)
gray_image = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)#convert the image to grayscale image

# smooth the graysacle image using gaussianblur method 
blur =cv2.GaussianBlur(gray_image, (5,5),0)
cv2.imshow("gray image",blur)
###############################################################################################################
print(image.shape)
height = image.shape[0]#get the hieght of image
width = image.shape[1]#get the width of image
#selecting the process area as a square
region_of_interest_vertices = [
    (0, height),
    (0 , height/2 +70),
    (width , height/2 +70),
    (width, height)
]
###################################################################################################################
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image =cv2.bitwise_and(img, mask)
    return masked_image
#####################################################################################################################

def drow_the_lines(img,lines):
    img=np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image,(x1,y1),(x2,y2),(0,255,0), thickness=3)

    img =cv2.addWeighted(img, 0.8, blank_image,1,1)
    return img
#####################################################################################################################

def display_lines(image, lines):
    image=np.copy(image)
    line_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    #line_image =np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image, (x1,y1),(x2,y2),(0,255,0),10)
    return line_image
####################################################################################################################
def make_coordinates(iamge ,line_paarameters):
    slope ,intercept = line_paarameters
    y1 = image.shape[0]
    y2 = int(y1*(2/5))
    x1 = int((y1 -intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])
#####################################################################################################################

def average_slope_itercept(image ,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 =line.reshape(4)
        parameters =np.polyfit((x1,x2), (y1,y2),1)
        #print(parameters)
        slope =parameters[0]
        intercept =parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope,intercept)) 
    left_fit_average =np.average(left_fit,axis=0)
    right_fit_average= np.average(right_fit, axis=0)
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image , right_fit_average)
    print(left_fit_average, 'left')
    print(right_fit_average, 'right')
    return np.array([left_line ,right_line])

####################################################################################################################
#gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
canny_image = cv2.Canny(gray_image, 50, 150)

cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

######################################################################################################################


lines =cv2.HoughLinesP(cropped_image, rho=10,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=100, maxLineGap=2)

#image_with_lines =drow_the_lines(image, lines)

averaged_lines =average_slope_itercept(lane_image,lines)
#image_with_lines =drow_the_lines(lane_image,averaged_lines)


#######################################################
line_image1 =display_lines(lane_image , averaged_lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image1,1,1)

########################################################
#sobel_image = cv2.Sobel(gray_image, 100,20)
plt.imshow(canny_image)
#cv2.imshow("result", cropped_image)
cv2.imshow("result1",combo_image)
#plt.imshow(image_with_lines)
plt.show()