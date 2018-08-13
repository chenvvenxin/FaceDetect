# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 16:08:07 2018

@author: chenvvenxin
"""

import cv2

face_cascade = cv2.CascadeClassifier("E:\\opencv\\opencv\\sources\\samples\\winrt_universal\\VideoCaptureXAML\\video_capture_xaml\\video_capture_xaml.Windows\\Assets\\haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier('E:\\opencv\\opencv\\sources\\data\\haarcascades\\haarcascade_mcs_eyepair_big.xml')
buct = cv2.imread("C:\\Users\\chenvvenxin\\Desktop\\buct.jpg")#buct的logo    

cap = cv2.VideoCapture(0)

while(True):
    
    success, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#之所以不直接在imread里以“灰度模式”打开是因为以彩色图片打开的方式更方便最后显示图片
    gray = cv2.equalizeHist(gray)#直方图均衡化    
    faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.1,minNeighbors = 5,minSize = (24, 24))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)#画矩形标注人脸范围
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,'Eye',(ex+x,ey+y), font, 0.5, (11,255,255), 1, cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'Face',(x,y), font, 0.5, (11,255,255), 1, cv2.LINE_AA)

    cv2.putText(img,"Chenvvenxin@outlook.com",(270,50),cv2.FONT_HERSHEY_PLAIN,1.5,(139,0,0),2)#在窗口中显示文字
    rows, cols = buct.shape[:2]#图像叠加，将Logo叠加到摄像头画面上去
    roi = img[:rows, :cols]    
    result = cv2.add(roi, buct)
    img[:rows, :cols] = result
    
    cv2.namedWindow('RealtimeFaceDetect', cv2.WINDOW_NORMAL)
    cv2.imshow("RealtimeFaceDetect", img)   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()