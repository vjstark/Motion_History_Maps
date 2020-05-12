import cv2
import numpy as np
import os
from scipy.spatial import distance
from scipy import ndimage
import math


def preprocess(motion_history):
    image_frame = motion_history
    (h, w) = image_frame.shape[:2]
    area_frame = h * w
    gray = image_frame
    _, thresh = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    img_erosion = cv2.erode(thresh, kernel, iterations=1)
    kernel2 = np.ones((6, 6), np.uint8)
    img_dilated = cv2.dilate(img_erosion, kernel2, iterations = 4)

    return img_dilated

def is_point_inside_rect(rect, x, y):
    x1 = rect[1].start
    x2 = rect[1].stop
    y1 = rect[0].start
    y2 = rect[0].stop

    if x >= x1 and x <= x2 and y >= y1 and y <= y2:
        return True

    return False


def filter_inside_rects(objects):
    slices = []
    print("filtered len ", len(objects))
    for i in range(len(objects)):
        aoi = objects[i]
        aoi_area = abs(aoi[0].stop - aoi[0].start) * (abs(aoi[1].stop - aoi[1].start))
        c = 0
        for j in range(len(objects)):
            if i != j:
                ob = objects[j]
                ob_area = abs(ob[0].stop - ob[0].start) * (abs(ob[1].stop - ob[1].start))
                x1 = ob[1].start
                x2 = ob[1].stop
                y1 = ob[0].start
                y2 = ob[0].stop
                if is_point_inside_rect(aoi, x1, y1) or is_point_inside_rect(aoi,x2,y1) or is_point_inside_rect(aoi,x1, y2) or is_point_inside_rect(aoi,x2,y2):
                    if aoi_area < ob_area:
                        c += 1

        print("Count", c)
        if c == 0:
            slices.append(aoi)

    return slices


def filter_small_objects(objects, frame_area, min_percentage=1 ):
    slices = []

    for ob in objects:
        rect_area = float(abs(ob[0].stop - ob[0].start) * abs(ob[1].stop - ob[1].start))
        print(rect_area, frame_area)
        percentage = 100.00 * rect_area / float(frame_area)
        print(f"percentage {percentage}")
        if percentage > min_percentage:
            slices.append(ob)

    return slices


def merge_closely_placed_blocks(slices):
    merged_slices = []
    for i in range(len(slices)):
        obj1 = slices[i]
        point_a = (obj1[1].start, obj1[0].start)
        point_b = (obj1[1].stop, obj1[0].start)
        point_c = (obj1[1].start, obj1[0].stop)
        point_d = (obj1[1].stop, obj1[0].stop)
        obj1_points = [point_a, point_b, point_c, point_d]
        for j in range(i+1, len(slices)):
            if i != j:
                obj2 =slices[j]
                point_a2 = (obj2[1].start, obj2[0].start)
                point_b2 = (obj2[1].stop, obj2[0].start)
                point_c2 = (obj2[1].start, obj2[0].stop)
                point_d2 = (obj2[1].stop, obj2[0].stop)
                obj2_points = [point_a2, point_b2, point_c2, point_d2]
                if check_proximity(obj1_points, obj2_points):
                    new_x1 = min(obj1[1].start, obj2[1].start)
                    new_y1 = min(obj1[0].start, obj2[0].start)
                    new_x2 = max(obj1[1].stop, obj2[1].stop)
                    new_y2 = max(obj1[0].stop, obj2[0].stop)
                    new_slice = (slice(new_y1, new_y2, None), slice(new_x1, new_x2, None))
                    merged_slices.append(new_slice)
                else:
                    merged_slices.append(obj1)


    print("asa",len(merged_slices))
    merged_slices = filter_inside_rects(merged_slices)

    return merged_slices


def check_proximity(a, b):
    for point1 in a:
        for point2 in b:
            dst = distance.euclidean(point1, point2)
            if dst < 50:
                return True
    return False


def add_padding(slices, frame_width, frame_height):

    padded_slices = []
    for i in range(0,len(slices)):
        obj = slices[i]
        x1 = obj[1].start
        x2 = obj[1].stop
        y1 = obj[0].start
        y2 = obj[0].stop
        width = abs(x2 - x1)
        height = abs(y2 -y1)
        margin = abs(height - width)
        padding = math.floor(margin/2)

        tmp_x1 = x1
        tmp_x2 = x2
        tmp_y1 = y1
        tmp_y2 = y2

        if width < height:
            tmp_x1 -= padding
            tmp_x2 += padding
            if tmp_x1 < 0:
                tmp_x1 = 0
            if tmp_x2 >= frame_width:
                tmp_x2 = frame_width - 1

        elif width > height:
            tmp_y1 -= padding
            tmp_y2 += padding
            if tmp_y1 < 0:
                tmp_y1 = 0
            if tmp_y2 >= frame_height:
                tmp_y2 = frame_height-1


        new_slice = (slice(tmp_y1, tmp_y2, None), slice(tmp_x1, tmp_x2, None))
        padded_slices.append(new_slice)

    return padded_slices

def correct_inccorect_points(obj, frame_heigh, frame_width):
    x1 =obj[1].start
    x2 = obj[1].stop
    y1 = obj[0].start
    y2= obj[0].stop
    if y1 < 0:
         y1 = 0
    if x1 < 0:
        x1 = 0
    if y2 >= frame_heigh:
        y2 = frame_heigh -1
    if x2 >= frame_width:
        x2 = frame_width -1
    new_slice = (slice(y1, y2, None), slice(x1, x2, None))
    return new_slice

def clean_objects(objects,h,w):
    slices = []
    for obj in objects:
        s = correct_inccorect_points(obj,h,w)
        slices.append(s)
    return slices



def findAOIs(motion_history):
    img = preprocess(motion_history)
    # Label objects
    labeled_image, num_features = ndimage.label(img)
    # Find the location of all objects
    slices = ndimage.find_objects(labeled_image)
    frame_area = motion_history.shape[0] * motion_history.shape[1]
    frame_width = motion_history.shape[1]
    frame_height = motion_history.shape[0]
    #merge closely paced objects

    print("No filtered ",len(slices))
    slices = merge_closely_placed_blocks(slices)
    #print("After close merge ",len(slices))
    ##filter small objects
    slices = filter_small_objects(slices, frame_area)
    #print("after small o filter",len(slices))
    ##filter by cutoff area
    slices = filter_inside_rects(slices)
    #print("filtered inside ",len(slices))
    ##add padding to rects to make it a square
    slices = add_padding(slices, frame_width, frame_height)
    slices = clean_objects(slices, frame_height, frame_width)
    return slices


# img = cv2.imread("frame3304.jpg", cv2.IMREAD_GRAYSCALE)
# objs = findAOIs(img)
# print("AOIS ", objs)
# c = 0
# for ob in objs:
#     c += 1
#     cv2.rectangle(img, (ob[1].start,ob[0].start), (ob[1].stop,ob[0].stop), (100,255,0), 1)
#
# cv2.imshow('Input', img)
# cv2.waitKey(0)
