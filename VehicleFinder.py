import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import time
import matplotlib.image as mpimg
from sklearn.externals import joblib
from utility import *


class VehicleFinder:

    def __init__(self, color_space='RGB', spatial_size=(32,32), hist_nbins=32, channel='all', hist_bins_range=(0,256)
                 , hog_orient=9, hog_pix_per_cell=8, hog_cell_per_block=8):
        self.color_space = color_space
        self.spatial_size = spatial_size
        self.hist_nbins = hist_nbins
        self.channel = channel
        self.hist_bins_range = hist_bins_range
        self.hog_orient = hog_orient
        self.hog_pix_per_cell = hog_pix_per_cell
        self.hog_cell_per_block = hog_cell_per_block
        self.clf = None
        self.scaler = None
        self.feature_vector_size = None

    def feature_extraction(self, img):
        """
        Extract features from single image
        :param img: RGB image of numpy array type, value range (0, 256)
        :return: hog of image
        """
        if self.color_space != 'RGB':
            feature_img = cvt_color(img, self.color_space)
        else:
            feature_img = np.copy(img)

        spatial_features = bin_spatial(feature_img, self.spatial_size)
        hist_features = color_hist(feature_img, self.hist_nbins, self.channel, self.hist_bins_range)

        # feature_img = feature_img.astype(np.float32) / 255
        if self.channel == 'all':
            hog_features = []
            for i_channel in range(feature_img.shape[2]):
                hog_features.append(get_hog(feature_img[:,:,i_channel], self.hog_orient, self.hog_pix_per_cell,
                                            self.hog_cell_per_block))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog(feature_img[..., self.channel], self.hog_orient, self.hog_pix_per_cell,
                                            self.hog_cell_per_block)

        feature_vec = np.concatenate([spatial_features, hist_features, hog_features])
        self.feature_vector_size = feature_vec.shape
        return feature_vec

    def feature_extraction_from_path(self, paths):
        """
        Extract features from images from designated path
        :param paths: list of img paths
        :return: feature vector
        """
        features = []
        for path in paths:
            img = mpimg.imread(path)
            if path.split('.')[-1] == 'png':
                img = img * 255
            features.append(self.feature_extraction(img))
        return features

    def train_from_path(self, car_paths, notcar_paths):
        car_features = self.feature_extraction_from_path(car_paths)
        notcar_features = self.feature_extraction_from_path(notcar_paths)

        X = np.vstack((car_features, notcar_features))
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
        self.feature_vector_size = X.shape[-1]
        X_scaler = StandardScaler().fit(X)
        X = X_scaler.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)

        clf = LinearSVC()
        t = time.time()
        clf.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'sec to train SVC')
        print('Test accuracy of SVC: {}'.format(round(clf.score(X_test, y_test), 4)))
        t = time.time()
        n_predict = 10
        print('Predicts: ', clf.predict(X_test[0:n_predict]))
        print('Truth: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2 - t, 5), 'sec to predict {} samples'.format(n_predict))

        self.scaler = X_scaler
        self.clf = clf

    def find_cars_from_img(self, img, ystart, ystop, scale):
        """
        
        :param img: RGB image to find cars from 
        :param ystart: 
        :param ystop: 
        :param scale: scale ratio. rescale image before searching for multiple size search windows!
        :return: 
        """
        draw_img = np.copy(img)
        img_search = img[ystart:ystop, ...]
        if self.color_space != 'RGB':
            feature_img = cvt_color(img_search, self.color_space)
        else:
            feature_img = np.copy(img_search)

        if scale != 1:
            scaled_shape = np.dot(feature_img.shape, scale).astype(np.int)
            feature_img = cv2.resize(feature_img, scaled_shape[1], scaled_shape[0])

        if self.channel == 'all':
            hogs = []
            for i_channel in range(feature_img.shape[2]):
                hogs.append(get_hog(feature_img[:,:,i_channel], self.hog_orient, self.hog_pix_per_cell,
                                            self.hog_cell_per_block, ravel=False))
        else:
            hogs = [get_hog(feature_img[..., self.channel], self.hog_orient, self.hog_pix_per_cell,
                                            self.hog_cell_per_block, ravel=False)]

        window_size = self.hog_pix_per_cell * self.hog_cell_per_block

        for x_window in range(hogs[0].shape[1]):
            for y_window in range(hogs[0].shape[0]):
                hog_features = []
                for ch_hog in hogs:
                    hog_features.extend(ch_hog[y_window:y_window+2, x_window:x_window+2].ravel())

                xleft = x_window * self.hog_pix_per_cell
                ytop = y_window * self.hog_pix_per_cell

                # extract image patch
                subimg = feature_img[ytop:ytop+window_size, xleft:xleft+window_size]

                # get color features
                spatial_features = bin_spatial(subimg, self.spatial_size)
                hist_feature = color_hist(subimg, self.hist_nbins, self.channel, self.hist_bins_range)

                print(np.array(spatial_features).shape, np.array(hist_feature).shape, np.array(hog_features).shape)
                import pdb; pdb.set_trace()
                test_features = self.scaler.transform(np.hstack((spatial_features, hist_feature, hog_features)))
                test_prediction = self.clf.predict(test_features)

                if test_prediction == 1:
                    box_left = np.int(xleft / scale)
                    box_top = np.int(ytop / scale)
                    win_size = np.int(window_size / scale)

                    cv2.rectangle(draw_img, (box_left, box_top+ystart), (box_left+win_size, box_top+ystart+win_size)
                                  ,color=(0,0,255), thickness=6)
        return draw_img
































































































