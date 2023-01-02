# ----------------------------- Include files
import mediapipe as mp
import cv2 as cv
import numpy as np
import math
import time
import handModule as hm

# ----------------------------- Global Variables

WIDTH = 1280
HEIGHT = 920

ptime = 0
ctime = 0

FINGERCOUNTER   = False
CENTEROFMASS    = False
CENTEROFPALM    = False
CHECKGRAB       = False
SWIPEDIRECTION  = False

# ----------------------------- OpenCV & Module initiation

cap = cv.VideoCapture(0)

detector = hm.HandDetector(detectionCon = 0.75, trackCon = 0.6)

detector.help()

while cap.isOpened():

    success, frame = cap.read()
    frame.flags.writeable = False
    # frame = cv.resize(frame, (frame.shape[0], HEIGHT));
    # frame = cv.flip(frame, 1)

    img = detector.findHands(frame)
    detector.findPosition(frame, 2)
    hasHands = detector.results.multi_hand_landmarks

    # img = cv.flip(img, 1)

    # -----------------------------FPS Calculations:
    ctime = time.time()
    fps = 1 / (ctime-ptime)
    ptime = ctime

    # -----------------------------Menu Display:
    cv.putText(img, "FPS: " + str(int(fps)), (50, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv.putText(img, "[G] Check Grab", (50, 100), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv.putText(img, "[F] Finger Count", (50, 150), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv.putText(img, "[M] Center of Mass (Red)", (50, 200), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv.putText(img, "[P] Palm Center (Blue)", (50, 250), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv.putText(img, "[D] Swipe Direction", (50, 300), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # -----------------------------Drawing Features:
    if FINGERCOUNTER == True:
        fCnt = detector.fingerCount(frame)
        cv.putText(img, "[Finger Count: " + str(int(len(fCnt))) + "]", (600, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    if CHECKGRAB == True:
        state = detector.checkGrabCnt(frame)
        cv.putText(img, "[Grabbing: " + str(int(state)) + "]", (1050, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    if CENTEROFMASS == True and hasHands:
        centers = detector.center_of_mass(frame, 0)
        cv.circle(img, (centers[0][0], centers[0][1]), 3, (0, 0, 255), 10)
    
    if CENTEROFPALM == True and hasHands:
        palm = detector.palm_center(frame, 0)
        cv.circle(img, (palm[0][0], palm[0][1]), 3, (255, 0, 0), 10)


    if SWIPEDIRECTION == True:
        direction = detector.swipeDirection(frame)
        cv.putText(img, "[Direction: " + detector.direction + "]", (1390, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        if direction != "None":
            detector.direction = direction
            print(direction)

    # -----------------------------Displaying
    cv.imshow('Video', img)

    # -----------------------------Toggling features for demonstration purposes.
    KEY = cv.waitKey(5)

    if KEY & 0xFF == ord('f'):
        FINGERCOUNTER = not FINGERCOUNTER

    if KEY & 0xFF == ord('g'):
        CHECKGRAB = not CHECKGRAB

    if KEY & 0xFF == ord('m'):
        CENTEROFMASS = not CENTEROFMASS

    if KEY & 0xFF == ord('p'):
        CENTEROFPALM = not CENTEROFPALM
    
    if KEY & 0xFF == ord('d'):
        SWIPEDIRECTION = not SWIPEDIRECTION
        detector.direction = "None"
        
    # -----------------------------Exit Condition
    if KEY == 27:     # ESC to break
        break

# ----------------------------- OpenCV release & destroy
cap.release()
cv.destroyAllWindows()