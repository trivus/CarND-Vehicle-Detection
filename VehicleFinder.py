from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import time
import matplotlib.image as mpimg
from collections import deque
from utility import *
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix


class VehicleFinder:

    def __init__(self, color_space='RGB', spatial_size=(32,32), hist_nbins=32, channel='all', hist_bins_range=(0,256)
                 , hog_orient=9, hog_pix_per_cell=8, hog_cell_per_block=8, heat_thresh=5, outfunc=None):
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
        self.heatmap = deque(maxlen=5)
        self.heat_thresh = heat_thresh
        self.outfunc = outfunc
        self.frame_count = 0

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
        hist_features = color_hist(feature_img, self.hist_nbins, hist_range=self.hist_bins_range)

        if self.channel == 'all':
            hog_features = []
            for i_channel in range(3):
                hog_channel = get_hog(feature_img[..., i_channel], self.hog_orient, self.hog_pix_per_cell,
                                            self.hog_cell_per_block)
                hog_features.append(hog_channel)
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
            if path.split('.')[-1] != 'png':
                img = img.astype(np.float) / 255
            features.append(self.feature_extraction(img))
        return features

    def train_from_path(self, car_paths, notcar_paths):
        car_features = self.feature_extraction_from_path(car_paths)
        notcar_features = self.feature_extraction_from_path(notcar_paths)

        X = np.vstack((car_features, notcar_features))
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
        self.feature_vector_size = X.shape[-1]
        #import pdb;pdb.set_trace()
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
        print('feature vector size: ', self.feature_vector_size)

        # train using whole dataset
        X, y = shuffle(X, y)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        print(confusion_matrix(y, y_pred))

        self.scaler = X_scaler
        self.clf = clf

    def find_cars_from_img(self, img, ystart, ystop,  scale, xstart=None, xstop=None, cell_per_step=2):
        """
        
        :param img: RGB image to find cars from 
        :param ystart: 
        :param ystop: 
        :param scale: scale ratio. rescale image before searching for multiple size search windows!
        :return: positive boxes
        """
        boxes = []
        img = img.astype(np.float32) / 255
        if xstart is None:
            xstart = 0
        if xstop is None:
            xstop = img.shape[1]
        img_search = img[ystart:ystop, xstart:xstop, :]

        if self.color_space != 'RGB':
            feature_img = cvt_color(img_search, self.color_space)
        else:
            feature_img = np.copy(img_search)

        if scale != 1:
            scaled_shape = np.dot(feature_img.shape, scale).astype(np.int)
            feature_img = cv2.resize(feature_img, (scaled_shape[1], scaled_shape[0]))

        if self.channel == 'all':
            hogs = []
            for i_channel in range(3):
                hogs.append(get_hog(feature_img[:,:,i_channel], self.hog_orient, self.hog_pix_per_cell,
                                            self.hog_cell_per_block, ravel=False))
        else:
            hogs = [get_hog(feature_img[..., self.channel], self.hog_orient, self.hog_pix_per_cell,
                                            self.hog_cell_per_block, ravel=False)]

        # size of train image
        window_size = 64
        nblocks_per_window = (window_size // self.hog_pix_per_cell) - self.hog_cell_per_block + 1
        cell_per_step = cell_per_step

        nxblocks = (feature_img[...,0].shape[1] // self.hog_pix_per_cell) - self.hog_cell_per_block + 1
        nyblocks = (feature_img[..., 0].shape[0] // self.hog_pix_per_cell) - self.hog_cell_per_block + 1
        nxsteps = (nxblocks - nblocks_per_window) // cell_per_step
        nysteps = (nyblocks - nblocks_per_window) // cell_per_step

        for xb in range(nxsteps):#range(0, hogs[0].shape[1], cell_per_step):
            for yb in range(nysteps):#range(0, hogs[0].shape[0], cell_per_step):
                x_window = xb*cell_per_step
                y_window = yb*cell_per_step

                hog_features = []
                for ch_hog in hogs:
                    hog_features.extend(ch_hog[y_window:y_window+nblocks_per_window,
                                        x_window:x_window+nblocks_per_window].ravel())

                xleft = x_window * self.hog_pix_per_cell
                ytop = y_window * self.hog_pix_per_cell

                # extract image patch
                subimg = feature_img[ytop:ytop+window_size, xleft:xleft+window_size]

                # get color features
                spatial_features = bin_spatial(subimg, self.spatial_size)
                hist_feature = color_hist(subimg, self.hist_nbins, hist_range=self.hist_bins_range)

                test_features = self.scaler.transform(
                    np.hstack((spatial_features, hist_feature, hog_features)).reshape(1,-1))
                test_prediction = self.clf.predict(test_features)

                if test_prediction == 1:
                    box_left = np.int(xleft / scale)
                    box_top = np.int(ytop / scale)
                    win_size = np.int(window_size / scale)

                    boxes.append(((box_left + xstart, box_top+ystart), (box_left+xstart+win_size, box_top+ystart+win_size)))

        return boxes

    def video_pipeline(self, image):
        # skip frames
        heatmap = np.zeros_like(image[..., 0])
        if self.frame_count % 3 == 0:
            # get heat map
            boxes1 = self.find_cars_from_img(image, 400, 600, .7)
            boxes2 = self.find_cars_from_img(image, 380, 480, 1.3, xstart=300, xstop=1100, cell_per_step=2)
            #boxes3 = self.find_cars_from_img(image, 350, 500, 1)
            boxes1.extend(boxes2)
            #boxes1.extend(boxes3)
            heatmap = apply_threshold(cal_heatmap(heatmap, boxes1), self.heat_thresh)
            self.heatmap.append(heatmap)

        summed_heatmap = np.zeros_like(heatmap).astype(np.float)
        for hmap in self.heatmap:
            summed_heatmap += hmap
        thresh_summed_heatmap = apply_threshold(summed_heatmap, 1)
        labeled_boxes = labeled_heat_boxes(thresh_summed_heatmap)

        if self.outfunc is None:
            out = np.copy(image)
        else:
            out = self.outfunc(image)
        out = draw_boxes(out, labeled_boxes)
        # heatmap debug image
        heatmap_debug = np.clip((summed_heatmap * 50), 0, 255).astype(np.uint8)
        out[-240:, 960:, :] = cv2.resize(cv2.cvtColor(heatmap_debug, cv2.COLOR_GRAY2RGB), (320, 240))
        self.frame_count += 1

        return out
