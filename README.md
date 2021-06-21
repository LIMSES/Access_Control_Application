# Human-Access-Control application using deep learning and object tracking algorithms
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
Before To get started, install the proper dependencies either via Anaconda or Pip.
