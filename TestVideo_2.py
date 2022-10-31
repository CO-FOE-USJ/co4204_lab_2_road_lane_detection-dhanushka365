import cv2
import numpy as np
import matplotlib.image as mpimg
import glob
#######################################################################################################################################
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])
#######################################################################################################################################
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])
#########################################################################################################################################
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny
########################################################################################################################################
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return line_image
########################################################################################################################################
def region_of_interest(image):
    height = image.shape[0]
    #polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    #polygons = np.array([[(0, image.shape[0]), (250, 290), (180, 320), (image.shape[1], image.shape[0])]])
    polygons = np.array([[(0, image.shape[0]),(0, image.shape[0]-100), (290, 320), (340, 320), (image.shape[1]-150, image.shape[0])]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def mark_lanes(lane_image):
    canny_image = canny(lane_image)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(lane_image, lines)
    line_image = display_lines(lane_image, averaged_lines)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    return combo_image

def createVideo():
    img_array = []
    for filename in glob.glob('Output/*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter('project2.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

paths = glob.glob('TestVideo_2/*.bmp')

for i,image_path in enumerate(paths):
    image = mpimg.imread(image_path)
    lane_image = np.copy(image)
    result = mark_lanes(lane_image)
    # plt.subplot(2,3,6)
    # plt.imshow(result)
    mpimg.imsave('Output_TestVideo_2/'+image_path[12:-4]+'_detected.jpg', result)

createVideo()