# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 16:08:07 2018

@author: chenvvenxin
"""

import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    cascade = cv2.CascadeClassifier("E:\\opencv\\opencv\\sources\\samples\\winrt_universal\\VideoCaptureXAML\\video_capture_xaml\\video_capture_xaml.Windows\\Assets\\haarcascade_frontalface_alt.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#转灰度图片
    gray = cv2.equalizeHist(gray)#直方图均衡化
    
    faces = cascade.detectMultiScale(gray,scaleFactor = 1.1,minNeighbors = 5,minSize = (24, 24))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)#画矩形标注人脸范围
    cv2.namedWindow('RealtimeFaceDetect', cv2.WINDOW_NORMAL)
    cv2.imshow("RealtimeFaceDetect", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()