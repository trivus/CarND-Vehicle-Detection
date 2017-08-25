import cv2
import numpy as np
from skimage.feature import hog
from scipy.ndimage.measurements import label

def bin_spatial(img, size):
    features = cv2.resize(img, size).ravel()
    return features


def color_hist(img, nbins, hist_range=(0,256)):
    chan1 = np.histogram(img[:, :, 0], bins=nbins, range=hist_range)
    chan2 = np.histogram(img[:, :, 1], bins=nbins, range=hist_range)
    chan3 = np.histogram(img[:, :, 2], bins=nbins, range=hist_range)
    hist_features = np.concatenate((chan1[0], chan2[0], chan3[0]))
    return hist_features


def get_hog(img, orient, pix_per_cell, cell_per_block, vis=False, ravel=True):
    """
    skimage hog function wrapper
    :param img: one channel image numpy array
    :param orient: 
    :param pix_per_cell: 
    :param cell_per_block: 
    :param vis: 
    :return: hog feature vector
    """
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=vis, feature_vector=ravel)
        return features, hog_image
        # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=vis, feature_vector=ravel)
        return features


def draw_boxes(img, boxes, color=(0, 0, 255), thick=6):
    """
    draw boxes on image
    :param img: 
    :param boxes: ((x1,y1), (x2,y2))
    :param color: 
    :param thick: 
    :return: 
    """
    imcopy = np.copy(img)
    for box in boxes:
        cv2.rectangle(imcopy, box[0], box[1], color, thick)
    return imcopy


def cvt_color(img, color_space):
    if color_space == 'HSV':
        feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif color_space == 'LUV':
        feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
        feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCrCb':
        feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        raise Exception('Unsupported color space.')
    return feature_img


def cal_heatmap(heatmap, box_list):
    """
    apply heatmap
    :param heatmap: img to draw heatmap on (zeros) 
    :param box_list: list of box coordinates ((x1,y1), (x2,y2))
    :return: heatmap img
    """
    if len(box_list) == 0:
        return heatmap
    for box in box_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    heatmap = np.clip(heatmap, 0, 255)
    return heatmap


def apply_threshold(img, threshold):
    """
    :param img: one channel 2d img 
    :param threshold: 
    :return: 
    """
    out = np.copy(img)
    out[out <= threshold] = 0
    return out

def labeled_heat_boxes(heatmap):
    """    
    :param heatmap: 
    :return: labeled boxes
    """
    labels = label(heatmap)
    box = []
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        box.append(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))
    return box


