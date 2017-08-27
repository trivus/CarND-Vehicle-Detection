import cv2
import numpy as np
from skimage.feature import hog
import glob
from scipy.ndimage.measurements import label
import os
from sklearn.externals import joblib

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
    out[out > threshold] = 1
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


def calibrate_from_dir(path, mtx_path='mtx.pkl', dist_path='dist.pkl'):
    """
    Calibrate camera with pictures in given directory.
    * File extension should be jpg.
    * Chessboard size 9 x 6
    :param path: path to dir containing chessboard images
    :param mtx_path: path to save mtx
    :param dist_path: path to save dist
    :return: mtx, dist
    """
    file_list = sorted(glob.glob(path+'*.jpg'))
    objpoints, imgpoints = [], []
    # world coordinate for chessboard corners (x, y, z)
    # (0,0,0), (1,0,0), (1,1,0), ...
    chess_obj_points = np.zeros((6*9, 3), np.float32)
    chess_obj_points[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    for file in file_list:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(img, (9, 6), None)
        if ret is True:
            objpoints.append(chess_obj_points)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    joblib.dump(mtx, mtx_path)
    joblib.dump(dist, dist_path)
    return mtx, dist


def get_birdview(img, m_path='m.pkl', mtx_path='mtx.pkl', dist_path='dist.pkl'):
    """
    get bird's view image from front camera image
    :param img: front camera image
    :param m_path: path to pickled M file
    :param mtx_path: path to pickled mtx file
    :param dist_path: path to pickled dist file
    :return: M, transformed image
    """
    mtx = joblib.load(mtx_path)
    dist = joblib.load(dist_path)
    d_img = cv2.undistort(img, mtx, dist, None, mtx)

    if os.path.isfile(m_path):
        M = joblib.load(m_path)
    else:
        # the coordinates were created by eye-inspecting sample pictures.
        #src = np.float32([[560, 477], [725, 477], [1035, 675], [271, 675]])
        #src = np.float32([[582, 462], [702, 462], [1035, 675], [271, 675]])
        src = np.float32([[559, 480], [734, 480], [1035, 675], [271, 675]])
        #dst = np.float32([[384, 0], [896, 0], [896, 720], [384, 720]])
        dst = np.float32([[300, 0], [900, 0], [900, 720], [300, 720]])
        M = cv2.getPerspectiveTransform(src, dst)
        joblib.dump(M, m_path)

    warped = cv2.warpPerspective(d_img, M, dsize=d_img.shape[1::-1])
    return M, warped


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    scale_factor = np.max(sobel) / 255
    sobel = (np.abs(sobel) / scale_factor).astype(np.uint8)
    grad_binary = np.zeros_like(sobel)
    grad_binary[(sobel > thresh[0]) & (sobel < thresh[1])] = 1
    return grad_binary


def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    dsobel = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(dsobel) / 255
    dsobel = (dsobel / scale_factor).astype(np.uint8)
    mag_binary = np.zeros_like(dsobel)
    mag_binary[(dsobel > thresh[0]) & (dsobel < thresh[1])] = 1
    return mag_binary


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    dsobel = np.arctan2(sobely, sobelx)
    dir_binary = np.zeros_like(dsobel)
    dir_binary[(dsobel > thresh[0]) & (dsobel < thresh[1])] = 1
    return dir_binary


def yellow_mask(img, sensitivity=100):
    yellow_lower = np.array([10, 0, 255-sensitivity])
    yellow_upper = np.array([40, 255, 255])
    return cv2.inRange(img, yellow_lower, yellow_upper) // 255


def white_mask(img, sensitivity=50):
    lower = np.array([0, 255 - sensitivity, 0])
    upper = np.array([255, 255, 255])
    return cv2.inRange(img, lower, upper) // 255
