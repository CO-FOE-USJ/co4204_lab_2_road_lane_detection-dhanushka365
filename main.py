#18_ENG_112
#importing some useful packages
from cv2 import line
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from os import listdir


####################################################################################################
image_folder = 'TestVideo_1'
video_name = "test1.avi"
images = [img for img in os.listdir(image_folder) if img.endswith(".bmp")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
#########################################################################################################

######################################################################################################################

def grayscale(img):
    """Applies the Grayscale transform"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#############################################################################################################    
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
###############################################################################################################
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[3]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#####################################################################################################################
def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # Make a copy of the original image.
    img = np.copy(img)
    
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    
    for line in lines:
        for x1,y1,x2,y2 in line:                             
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
            
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    
    # Return the modified image.
    return img

###################################################################################################################
# Return a thresholded image using sobel magnitude and angle
def sobel_threshold(img, mag_thresh = (30,255),ang_thresh = (.8, 1.4)):
    
    # Apply x and y gradient with the OpenCV Sobel() function and take abs value
    x_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    y_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    
    # absolute value of direction of gradient
    grad_ang = np.arctan2(y_sobel, x_sobel)
    
    # magnitude of direction of gradient
    grad_mag = np.sqrt(y_sobel**2 + x_sobel**2)
    scale_factor = np.max(grad_mag)/255 
    grad_mag = (grad_mag/scale_factor).astype(np.uint8) 
    
    # Create a copy of sobel
    # Sobel is 1 px smaller on all sides so img will not work here
    binary_output = np.zeros_like(x_sobel).astype(np.uint8)
    
    # Apply Gradient
    #binary_output[(grad_mag >= mag_thresh[0]) & (grad_mag <= mag_thresh[1]) &
    #            (abs_grad_dir >= ang_thresh[0]) & (abs_grad_dir <= ang_thresh[1])] = 1
    binary_output[
        (grad_mag > mag_thresh[0]) & 
        (grad_mag < mag_thresh[1]) &
        (grad_ang > ang_thresh[0]) & 
        (grad_ang < ang_thresh[1])
    ] = 255

    return binary_output
###################################################################################################################
def show_example_sobel(img):
    
    #hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS).astype(np.float)
    #rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #gray scale image conversion
    gray = grayscale(img)
    #Gaussian blur
    kernel_size = 5
    blur_gray = gaussian_blur(gray,kernel_size)
    saturation_sobel_img = sobel_threshold(gray)#,mag_thresh,ang_thresh)
    print(saturation_sobel_img)
    return saturation_sobel_img
    

###########################################################################################################
def make_coordinates(iamge ,line_paarameters):
    slope ,intercept = line_paarameters
    y1 = image.shape[0]
    y2 = int(y1*(2.9/5))
    x1 = int((y1 -intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])
#############################################################################################################
def average_slope_itercept(image ,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 =line.reshape(4)
        parameters =np.polyfit((x1,x2), (y1,y2),1)
        #print(parameters)
        slope =parameters[0]
        intercept =parameters[1]
        if (slope <= -0.5 and slope >= -1.5):
            left_fit.append((slope, intercept))
        elif(slope >= 0.5 and slope <= 1.5):
            right_fit.append((slope,intercept)) 
    left_fit_average =np.average(left_fit,axis=0)
    right_fit_average= np.average(right_fit, axis=0)
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image , right_fit_average)
    print(left_fit_average, 'left')
    print(right_fit_average, 'right')
    return np.array([left_line ,right_line])

##############################################################################################################
def display_lines(image, lines):
    image=np.copy(image)
    line_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    #line_image =np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image, (x1,y1),(x2,y2),(255,0,0),3)
    return line_image

###############################################################################################################

#############################################################################################################
def average_slope_itercept1(image ,lines):
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if (math.fabs(slope) < 0.5 and math.fabs(slope) > -0.5):
                continue
            if (slope <= -0.5 and slope >= -1.5):
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            elif(slope >= 0.5 and slope <= 1.5):
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])
            
    min_y = int(image.shape[0] * (2 / 5))
    max_y = int(image.shape[0])

    poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))

    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))

    poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))

    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))
    #print(poly_left)
    x1= left_x_start
    y1= max_y
    x2= left_x_end
    y2= min_y
    x3 = right_x_start
    y3 = max_y
    x4=right_x_end
    y4=min_y
    u = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
    #print(u)
    x = x1 + u * (x2-x1)
    y = y1 + u * (y2-y1)
    #print(x,y)
    line_image = draw_lines(
            image,
            [[
                [left_x_start, max_y, left_x_end, min_y],
                [right_x_start, max_y, right_x_end, min_y],
            ]],
            thickness=5,
            )
    circle_image =cv2.circle(line_image, (int(x),int(y)), 10,(255,255,0),3 )
    final_image = cv2.addWeighted(line_image, 1, circle_image,0.1,1)     
    plt.figure()
    plt.imshow(final_image)
    plt.show()       
############################################################################################################


################################################################################################################
def process_image1(image):
    image_path = "images"
    #gray scale image conversion
    gray = grayscale(image)

    #Gaussian blur
    kernel_size = 5
    blur_gray = gaussian_blur(gray,kernel_size)

    edges = show_example_sobel(lane_image)


    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    height = image.shape[0]#get the hieght of image
    width = image.shape[1]#get the width of image
    region_of_interest_vertices = [
        (0, height),
        (0 , height/2 +50),
        (width , height/2 +50),
        (width, height)
    ]

    vertices = np.array([region_of_interest_vertices], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    #plt.imshow(masked_edges)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 10 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 100    # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 1 # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    #print(lines)


    #average_slope_itercept(image, lines)
    averaged_lines =average_slope_itercept(image,lines)
    print(averaged_lines)
    line_image1 =display_lines(image , averaged_lines)
    x1 = averaged_lines[0][0]
    y1 = averaged_lines[0][1]
    x2 = averaged_lines[0][2]
    y2 = averaged_lines[0][3]
    x3 = averaged_lines[1][0]
    y3 = averaged_lines[1][1]
    x4 = averaged_lines[1][2]
    y4 = averaged_lines[1][3]

    u = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
    print(u)
    x = x1 + u * (x2-x1)
    y = y1 + u * (y2-y1)
    print(x,y)
    combo_image = cv2.addWeighted(image, 0.8, line_image1,1,1)

    cv2.circle(combo_image, (int(x),int(y)), 10,(255,255,0),3 )
    cv2.imshow("line result", combo_image)
    #cv2.imshow('result',cropped_image)
    cv2.waitKey(0) #display result window infinetly untill we press anything from keyboard

##############################################################################################################################
def process_image(image):
    image_path = "images"
    #gray scale image conversion
    gray = grayscale(image)

    #Gaussian blur
    kernel_size = 5
    blur_gray = gaussian_blur(gray,kernel_size)

    edges = show_example_sobel(lane_image)


    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    height = image.shape[0]#get the hieght of image
    width = image.shape[1]#get the width of image
    region_of_interest_vertices = [
        (0, height),
        (0 , height/2 +50),
        (width , height/2 +50),
        (width, height)
    ]

    vertices = np.array([region_of_interest_vertices], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    #plt.imshow(masked_edges)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 100    # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 1 # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    #print(lines)


    average_slope_itercept1(image, lines)

##############################################################################################################################
# get the path or directory
folder_dir = "/home/uduwela/co4204_lab_2_road_lane_detection-dhanushka365/TestVideo_1"
for images in os.listdir(folder_dir):
 
    # check if the image ends with png or jpg or jpeg
    if (images.endswith(".bmp")):
        # display
        a= folder_dir +"/"+ images
        #reading in an image
        image = mpimg.imread('/home/uduwela/co4204_lab_2_road_lane_detection-dhanushka365/TestVideo_1/Right_18.bmp')
        lane_image = np.copy(image)
        #printing out some stats and plotting
        #print('This image is:', type(image), 'with dimensions:', image.shape)
        #plt.imshow(image)  
        #plt.show()
        process_image(lane_image)
        #process_image1(lane_image)



        # Try to work with sobel operator but wasn't able to to identify the corrrect treshhold values .therefore the function gives a weeka edge image at last.