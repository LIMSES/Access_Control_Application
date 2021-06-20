import cv2 
import numpy as np

cap1 = cv2.VideoCapture('./data/video/frontdoor.mp4')
if not cap1.isOpened():
    print("Video1 open failed!")
w1 = round(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h1 = round(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps1 = cap1.get(cv2.CAP_PROP_FPS)
fourcc1 = cv2.VideoWriter_fourcc(* 'DIVX')

cap2 = cv2.VideoCapture('./data/video/backdoor.mp4')
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