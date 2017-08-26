# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/train_imgs_cars.png
[image2]: ./output_images/train_imgs_notcars.jpg
[image3]: ./output_images/car_hog.jpg
[image4]: ./output_images/notcar_hog.jpg
[image5]: ./output_images/detector_heatmap.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in get_hog() function, lines 22 through 42 of the file `utility.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(8, 8)`:

![alt text][image3]
![alt text][image4]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters that produced best results were: `orientation=9, pixels_per_cell=8 and cell_per_block=2`, with all three channels.  
Using less channels made the detector faster, but at cost of accuracy.  

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Train pipeline can be found at line 72 to 99 in `VehicleFinder.py`. I used template matching, color histogram and HOG features.  
The training data was shuffled and divided into training/validation set using sklearn library. The feature values were normalized using `StandardScaler` from sklearn.  
The accuracy on the validation set was .9904 and feature vector size was 6156.  

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used HOG subsampling technique, which can be found at line 101 to 173 in `VehicleFinder.py`.  
As cars further away will be displayed smaller at upper portion of the image, I used two scales of windows.  
Bigger window on bottom portion of the image (y-coordination 400 to 600) and smaller window on upper portion of the image (380 to 480).  
Exact scales were tuned by visual inspection.  

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I tested several parameters. The parameters that produced the best result were:  
{color_space='YCrCb', spatial_size=(16,16), hist_nbins=32, hist_bins_range=(0,1.0),  
hog_orient=9, hog_pix_per_cell=8, hog_cell_per_block=2, channel='all', heat_thresh=4)}  

![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/cgmf2Wnv6PU)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. The code can be found at line 175 to 196 in `VehicleFinder.py`.  



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline worked reasonably well on the project video, but it includes false positives, and fails to detect a car for couple of frames. I could add more smoothing function to make it more robust,  
but doing so may negatively affect accuracy and responsiveness. I would like to try out CNN method to see if works better than hand-crafted features + LVC. 
 

