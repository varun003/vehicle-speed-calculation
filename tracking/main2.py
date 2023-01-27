import cv2
from tracker import *
import numpy as np

# creating tracker
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture('highway.mp4')

object_detector = cv2.createBackgroundSubtractorMOG2(history=200,varThreshold=100)

while True:
    ret,frame = cap.read()
    height,width,_ = frame.shape
    # print(height,width)

    roi = frame[350:600,500:800] 
    # cv2.rectangle(frame, (500,350), (800,600), (0, 255, 0), 2)
    pts = np.array([[580,350],[770,350],[800,600],[500,600]],np.int32)     
    ## (867,587),(1126,581),(826,732),(1150,748)

    pts = pts.reshape((-1, 1, 2))
    isClosed = True
    color = (255, 0, 0)
    thickness = 2

    cv2.polylines(frame,[pts],isClosed,color,thickness)

    # roi = cv2.rectangle(frame,pt1=(800,300),pt2 = (500,800),color = (255,0,0),thickness = 2)

    # object detection

    mask = object_detector.apply(roi)  # since we want to detect objects within ROI,so we'll replace "frame" with "roi" region
    _,mask = cv2.threshold(mask,254,255,cv2.THRESH_BINARY)

    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        # calculate area and remove all small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            # cv2.drawContours(roi,[cnt],-1,(0,255,0),2)  # frame is replaced with roi
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
            # print(x,y,w,h)
            detections.append([x,y,w,h])

    ## object tracking

    boxes_id = tracker.update(detections)
    for box_id in boxes_id:
        x,y,w,h,id = box_id
        cv2.putText(roi,str(id),(x,y-15),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.rectangle(roi, (x,y) , (x + w,y + h),(0,255,0),2)
    print(boxes_id)

    print(detections)
    # cv2.imshow('roi',roi)
    cv2.imshow('frame',frame) 

    key = cv2.waitKey(30)
    if key == ord('q'):

        break

cap.release()
cv2.destroyAllWindows