# Tutorial: Human-Access-Control application using deep learning and object tracking algorithms
**Author:** S.-S. Lim & H.-J. Kim<br>
**Date created:** 2021/06/21<br>

## Introduction
Since the outbreak of COVID-19, people have been restricted from entering various places. However, automation of access control is not yet used in most places.
So, this application aims to automatically identify and control people's access to rooms, buildings, or outdoors.
This application can be used for personnel control due to government guidelines, attendance checks in classes, building usage status, etc.<br><br>
The application was created using open source YOLOv4, Deep SORT and TensorFlow.
YOLOv4 is an algorithm that performs object detection using deep convolutional neural networks, and is created by adding additional algorithms to YOLOv3.
YOLOv4 is a 10% improvement in accuracy (AP) over YOLOv3 and has the advantage of fast and accurate object detection in a typical learning environment using only one GPU.
Deep SORT(Simple Online and Realtime Tracking with a Deep Association Metric) is an algorithm for object tracking.
It was created using deep learning, Kalman filter and Hungarian algorithm.<br>

**References:**
  * [YOLOv4 and Deep SORT: https://github.com/theAIGuysCode/yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort)
  * [Counting the passing objects: https://github.com/emasterclassacademy/Single-Multiple-Custom-Object-Detection-and-Tracking](https://github.com/emasterclassacademy/Single-Multiple-Custom-Object-Detection-and-Tracking)

## Requirements
To get started, you must first download the YOLOv4 and Deep SORT algorithm codes and directories from my GitHub repository. Download all files: https://github.com/LIMSES/Access_Control_Application/<br>

Also, you need at least one GPU to use YOLOv4. CUDA toolkit must be installed for GPU use.
CUDA Toolkit version 10.1 is the proper version for the TensorFlow version used in this application.
https://developer.nvidia.com/cuda-10.1-download-archive-update2<br>

After that, install the proper dependencies either via Anaconda. You can install dependencies through either conda-gpu.yml or requirements-gpu.txt.
### Tensorflow GPU
```bash
# Using conda-gpu.yml
conda env create -f conda-gpu.yml
conda activate yolov4-gpu

# or you can use requirements-gpu.txt
pip install -r requirements-gpu.txt
```
## Training Data
For object detection, the application must first be trained with a large number of data.
However, since the application aims to detect only humans, it use an official pre-trained YOLOv4 model that is able to detect 80 classes without additional training data.
Download pre-trained yolov4.weights file: https://drive.google.com/file/d/1VCRU3SpO5x76KngBr8FHAGR68UqMgbEs/view?usp=sharing <br>

You must download and extract the weights.zip file and place the yolov4.weights file in the data directory of your workspace(./data/yolov4.weights).
If the download doesn't work properly, refer to the [reference](https://github.com/theAIGuysCode/yolov4-deepsort) in this tutorial.<br>

The weight file downloaded is a darknet weight file. Transformation is required to apply file to tensorflow models.
If you run the save_model.py, checkpoints directory is created and tensorflow model is stored.
### Convert darknet weights to tensorflow model
```bash
python save_model.py --model yolov4
```
## Videos
In a place with one entrance, the application can be operated based on real-time images of that one entrance.
Likewise, in another place with two or more entrances, the application should be operated based on two or more images.
Simultaneous processing can be done through communication, such as ROS systems, but this tutorial combines images for easy processing.<br>

The videos going to use in the tutorial are entrance to the front and back door before the start of DLIP class at Handong Global University.
You can download and extract the videos.zip file and place the videos directory in the data directory(./data/videos/frontdoor.mp4).
Download sample videos: https://drive.google.com/file/d/10Cts1ObT_e_8B6jleXzUCb2PsgGcoMiM/view?usp=sharing <br>

### Front door
<p align="center"><img src="data/helpers/frontdoor.JPG" width="400" height="240"\></p>

### Back door
<p align="center"><img src="data/helpers/backdoor.JPG" width="400" height="240"\></p>

### Combining two 
```bash
import cv2 
import numpy as np

cap1 = cv2.VideoCapture('./data/videos/frontdoor.mp4')
if not cap1.isOpened():
    print("Video1 open failed!")
w1 = round(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h1 = round(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps1 = cap1.get(cv2.CAP_PROP_FPS)
fourcc1 = cv2.VideoWriter_fourcc(* 'DIVX')

cap2 = cv2.VideoCapture('./data/videos/backdoor.mp4')
if not cap2.isOpened():
    print("Video2 open failed!")
w2 = round(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
h2 = round(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps2 = cap2.get(cv2.CAP_PROP_FPS)
fourcc2 = cv2.VideoWriter_fourcc(* 'XVID')

size = (1020, 1020)
add_size = (2040, 1020)
writer = cv2.VideoWriter('demo2.mp4', fourcc1, fps1, add_size)
currentFrame = 0

while(True):
    return_value1, leftimg = cap1.read()
    return_value2, rightimg = cap2.read()

    if return_value1 == False:
        break
    if return_value2 == False:
        break
    # if currentFrame == 2:   #if you want to drop frame
    #     currentFrame = 0
    # if currentFrame == 0:

    leftimg = cv2.resize(leftimg, size)
    rightimg = cv2.resize(rightimg, size)
    add_img = np.hstack((leftimg, rightimg))
    cv2.imshow("Color", add_img)
    writer.write(add_img)
    if cv2.waitKey(1)&0xFF ==27:
        break
    #currentFrame += 1

cap1.release()
cap2.release()
cv2.destroyAllWindows()
```
The videos are combined for simultaneous identification.






