

import cv2
import numpy as np
from scipy.spatial import distance as dist 


BINARY_THRESHOLD_BLACK = 32
CAMERA_WIDTH = 1024
CAMERA_HEIGHT = 768
BIRD_WIDTH = 550
BIRD_HEIGHT = 350
CAMERA_CORNERS = [[0, 0], [CAMERA_WIDTH, 0], [CAMERA_WIDTH, CAMERA_HEIGHT], [0, CAMERA_HEIGHT]]
BIRD_CORNERS = [[0, 0], [BIRD_WIDTH, 0], [BIRD_WIDTH, BIRD_HEIGHT], [0, BIRD_HEIGHT]]
MAP_WIDTH = 1200
MAP_HEIGHT = 900

RES_THRESHOLD_BLACK = 0.83
OVERLAP_THRESHOLD = 0



def filter_grayscale(img):
	"given image in bgr colorspace, convert it to grayscale then convert it to binary"
	#convert to grayscale both image and template
	img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#convert to binary both image and template
	threshold = BINARY_THRESHOLD_BLACK
	ret, image_binary = cv2.threshold(img_grayscale, threshold, 255, cv2.THRESH_BINARY)

	return image_binary
	

def non_max_suppression_fast(boxes, overlapThresh):
	"""Makes sure to only have one set of coordinate per cross to find the corners of the map"""
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")


#----------------------------------------------------------------------------------------------------------------------------------


def order_points(pts):
	"""Orders points of rectangle so that they are in the order : top_left, top_right, bottom_right, bottom_left """
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")


#--------------------------------------------------------------------------------------------------------------------------------------------------


def find_4_corners(img, template, method):
	"""Finds the coordiantes of the corners of the map; 	
	arguments :	
	| img : image in the bgr color space;	
	| template : image that has been filtered and is in the binary color space;	
	| method : cross-correlation method for template matching
	"""
	is_found = False
	
	#filter the image
	filtered_img = cv2.GaussianBlur(img,(5,5), 1)
	
	#convert image and template to binary
	image_binary = filter_grayscale(filtered_img)
	template_binary = filter_grayscale(template)
	
	cv2.imshow("binary of black", image_binary)
	#get template dimensions
	c, w, h  = template.shape[::-1]
	#print(template.shape[::-1])
	
	#match template to image
	res = cv2.matchTemplate(image_binary,template_binary,method)
	(yCoords, xCoords) = np.where(res >= RES_THRESHOLD_BLACK)


	# initialize our list of rectangles
	rects = []
	# loop over the starting (x, y)-coordinates again
	for (x, y) in zip(xCoords, yCoords):
		# update our list of rectangles
		rects.append((x, y, x + w, y + h))
	
	# apply non-maxima suppression to the rectangles
	pick = non_max_suppression_fast(np.array(rects), OVERLAP_THRESHOLD)
	#print("[INFO] {} matched locations *after* NMS".format(len(pick)))
	
	#create a list of centers of the rectangles
	cross_points = []
	for (startX, startY, endX, endY) in pick:
	    # draw the bounding box on the image
		cv2.rectangle(image_binary, (startX, startY), (endX, endY), (0, 0, 255), 3)
		cross_points.append([int((startX+endX)/2), int((startY+ endY)/2)]) 

	cv2.imshow("binary of black", image_binary)
	#cv2.imshow("template", template_binary)

	if (len(pick) == 4) :
		is_found = True
	
	cross_points = np.array(cross_points)
	cv2.imwrite("template_cross_binary.png", image_binary)
	
	return is_found, cross_points



#----------------------------------------------------------------------------------------------------------------------------------


def warp_image(image, template, method, old_corners):
	"""Transforms the camera image to a map bird's eye view with only the map"""
	is_found, cross_points = find_4_corners(image, template, method)
	
	if is_found:
		ordered_crosses = order_points(cross_points)
		#store corners positions in case we have a frame where we don't see them anymore
		old_corners = ordered_crosses
	else : 
		#will become the last found corners, 
		ordered_crosses = old_corners

	ordered_crosses = np.float32(ordered_crosses)
	bird_corners = np.float32(BIRD_CORNERS)

	matrix = cv2.getPerspectiveTransform(ordered_crosses, bird_corners)
	bird_image = cv2.warpPerspective(image, matrix, (BIRD_WIDTH, BIRD_HEIGHT))

	return is_found, bird_image, old_corners








# Open the default camera
cam = cv2.VideoCapture(1)

if not cam.isOpened():
    print("Error: Could not open camera.")

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

template_cross = cv2.imread('/home/victor-pdt/Documents/Mobile_Robotics/template_cross_dll.jpg')
method = eval('cv2.TM_CCORR_NORMED')


while True:
    
    ret, image = cam.read()
    cv2.imshow("camera", image)

    grayscale = filter_grayscale(image)
	
    is_found, cross_points = find_4_corners(image, template_cross, method)
    print(len(cross_points))

    #cv2.imshow("grayscale", grayscale)
    
	# Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        cv2.imwrite("crosses_setup.jpg", grayscale)
        break
	
cam.release()
cv2.destroyAllWindows()