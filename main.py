#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

#########################################################################################################
#reading in an image
image = mpimg.imread('/home/uduwela/Project-Finding-Lane-Lines-on-the-Road-master/TestVideo_1/Right_22.bmp')
lane_image = np.copy(image)
#printing out some stats and plotting
#print('This image is:', type(image), 'with dimensions:', image.shape)
#plt.imshow(image)  
#plt.show()
######################################################################################################################

def grayscale(img):
    """Applies the Grayscale transform"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


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

####################################################################################################################
#gray scale image conversion
gray = grayscale(image)
#####################################################################################################################
#Gaussian blur
kernel_size = 5
blur_gray = gaussian_blur(gray,kernel_size)
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
    
###################################################################################################################
edges = show_example_sobel(lane_image)
#####################################################################################################################
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
################################################################################################################
vertices = np.array([region_of_interest_vertices], dtype=np.int32)
masked_edges = region_of_interest(edges, vertices)
#plt.imshow(masked_edges)

##############################################################################################################
# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 10 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 70    # minimum number of votes (intersections in Hough grid cell)
min_line_length = 70 #minimum number of pixels making up a line
max_line_gap = 1 # maximum gap in pixels between connectable line segments

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
#print(lines)
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


line_image = draw_lines(
        image,
        [[
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y],
        ]],
        thickness=5,
        )

plt.figure()
plt.imshow(line_image)
plt.show()

################################################################################################################