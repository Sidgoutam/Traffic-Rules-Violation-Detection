# Traffic-Rules-Violation-Detection
A deep learning based technique to detect the traffic rules violation
There are several Neural Network architectures for detection:-
   1. R-CNN family of architectures
   2. Single Shot Detectors
   3. YOLO — You Only Look Once
   
# Idea

Two of the many traffic rules violation have been taken into focus in this project
1. Helmet detection
2. Multiple rider detection

# Aim

In this project we take an input video and firstly break it into frames. Process those images into our trained model which has a positive set and negative set of images. After the images are processed we get an output video which detects the helmets on a person.(Object detection)

Secondly, for detection of multiple rider on a single vehicle we use another deep learning method that gives an identity number to each of the person after detecting them throughout the video. That is how we keep track of every person and detect multiple riders. (Object tracking)
   
# Implementation

Let’s begin by importing the necessary libraries. The OpenCV library is going to be our best friend for this project as it has several helpful functions for manipulating images as well as useful modules such as ‘dnn’.
Since we’ll be using a pre-trained model, we’d have to download certain files. The “weights” file , the “configuration” file, and the “coco-names” file. The weights and the configuration file can be found in this link https://pjreddie.com/darknet/yolo/ and the coco-names file can be downloaded/copied from https://github.com/pjreddie/darknet/blob/master/data/coco.names. There are several pre-trained models available and we would be using the “YOLOv3–416" model. The models are trained on the MS COCO dataset which has 80 classes of objects present in it.

The image is then given to the model and a forward-pass is performed. The output of which gives us a list of detections. From this list a set of bounding-box co-ordinates for each object detected is obtained as shown below. We use a confidence threshold value to filter out weak detections. The default value I’ve used for confidence threshold is ‘0.5’. All the bounding-box co-ordinates, their class-ids and their corresponding confidence values are stored in lists “boxes”, “class_ids ”and “confidences” respectively.

Now that we have obtained the locations of objects in our image, it’s time to sketch their bounding-box and tag them. The draw_boxes() function does this for us. One problem that we might encounter in our journey is that, the objects, sometimes, may be detected more than once. To avoid such a scenario, we’ll employ Non-Maximum Suppression (aka Non-Maxima Suppression). The default value I’ve used for NMS threshold is ‘0.4’. This is what is performed by the cv2.dnn.NMSBoxes() function down below. We finally display the output image using the cv2.imshow() function.

# Object tracking is the process of:
  1. Taking an initial set of object detections (such as an input set of bounding box coordinates)
  2. Creating a unique ID for each of the initial detections
  3. And then tracking each of the objects as they move around frames in a video, maintaining the assignment of unique IDs
  
# Multiple object tracking
In this type of tracking, we are expected to lock onto every single object in the frame, uniquely identify each one of them and track all of them until they leave the frame.

# Kalman Filters
In almost any engineering problem that involves prediction in a temporal or time series sense, be it computer vision, guidance, navigation or even economics, “Kalman Filter” is the go to algorithm. Kalman filter is a crucial component in deep SORT. Our state contains 8 variables; (u,v,a,h,u’,v’,a’,h’) where (u,v) are centres of the bounding boxes, a is the aspect ratio and h, the height of the image. The other variables are the respective velocities of the variables.

As we discussed previously, the variables have only absolute position and velocity factors, since we are assuming a simple linear velocity model. The Kalman filter helps us factor in the noise in detection and uses prior state in predicting a good fit for bounding boxes.

For each detection, we create a “Track”, that has all the necessary state information. It also has a parameter to track and delete tracks that had their last successful detection long back, as those objects would have left the scene.

Also, to eliminate duplicate tracks, there is a minimum number of detections threshold for the first few frames.

1. The distance metric
2. The efficient algorithm
3. The appearance feature vector
