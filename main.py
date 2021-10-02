import mediapipe as mp
import cv2 as cv
import numpy as np
import math
import time
import handModule as hm

####################################

WIDTH = 1280
HEIGHT = 920

ptime = 0
ctime = 0

####################################

cap = cv.VideoCapture(0)

detector = hm.HandDetector(detectionCon = 0.75, trackCon = 0.6)

detector.help()

while cap.isOpened():

    success, frame = cap.read()
    
    frame = cv.resize(frame, (WIDTH, HEIGHT));

    img = detector.findHands(frame)
    detector.findPosition(frame)
    # img = cv.flip(img, 1)

    fCnt = detector.fingerCount(frame)

    ctime = time.time()
    fps = 1 / (ctime-ptime)
    ptime = ctime

    cv.putText(img, "FPS: " + str(int(fps)), (50, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv.putText(img, "Finger Count: " + str(int(len(fCnt))), (650, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv.imshow('Video', img)

    if cv.waitKey(1) == 27:     # ESC to break
        break