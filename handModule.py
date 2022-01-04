import mediapipe as mp
import cv2 as cv
import numpy as np
import math
import time



class HandDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon= 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
                                        self.mode,
                                        self.maxHands,
                                        self.detectionCon,
                                        self.trackCon
                                        )
        self.mp_draw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(207,252,3), thickness=5, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(255,255,255), thickness=5, circle_radius=1)
                    )
            centers = self.center_of_mass(img, 0)
            palm = self.palm_center(img, 0)
            # Center of mass and palm center
            cv.circle(img, (centers[0][0], centers[0][1]), 3, (0, 0, 255), 5)
            cv.circle(img, (palm[0][0], palm[0][1]), 3, (255, 0, 0), 5)

        return img
    
    def findPosition(self, img, handNo = 0, draw = True):
        self.retList = []

        if handNo == 2:
            if self.results.multi_hand_landmarks:
                for hands in self.results.multi_hand_landmarks:
                    for ind, lm in enumerate(hands.landmark):
                        h, w, c = img.shape
                        # h, w = 1, 1
                        cx, cy = int(w*lm.x), int(h*lm.y)
                        self.retList.append([ind, cx, cy])
                    if draw:
                        cv.putText(img, "Both Hands", (50, 70), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)

        else:
            if self.results.multi_hand_landmarks:
                currentHand = self.results.multi_hand_landmarks[handNo]
                for ind, lm in enumerate(currentHand.landmark):
                    h, w, c = img.shape
                    # h, w = 1, 1
                    cx, cy = int(w*lm.x), int(h*lm.y)
                    self.retList.append([ind, cx, cy])

                    if draw:
                        cv.putText(img, ("Hand: " + str(handNo)),(50, 70), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
            
        return self.retList

    def fingerCount(self, img, handNo = 0, draw = True):
        # index: 8 = tip, 5 = knuckle
        # middle: 12 = tip, 9 = knuckle
        total = 0
        indexList = []
        if self.results.multi_hand_landmarks:
            center = self.center_of_mass(img, handNo)    
            palm_c = self.palm_center(img, handNo)

            for i in range(8, 24, 4):
                distA = self.dist(palm_c[0][0], palm_c[0][1], self.retList[i][1], self.retList[i][2])
                distB = self.dist(palm_c[0][0], palm_c[0][1], self.retList[i-2][1], self.retList[i-2][2])
                # cv.circle(img, (self.retList[i][1], self.retList[i][2]), 10, (255, 255, 255), 10)
                
                print(i," ",distA, " " ,distB)
                if distA > distB:
                    indexList.append(i)
                    total+=1

            distA = self.dist(palm_c[0][0], palm_c[0][1], self.retList[4][1], self.retList[4][2])
            distB = self.dist(palm_c[0][0], palm_c[0][1], self.retList[2][1], self.retList[2][2])
            
            if (distA > distB):
                indexList.append(4)
                total+=1

        return indexList

    def checkGrab(self, img, handNo = 0, draw = True):
        avg_x = 0
        avg_y = 0
        length = img.shape[1]
        width = img.shape[0]

        finger_tips = []

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                finger_tips.append(hand_landmarks.landmark[4])
                finger_tips.append(hand_landmarks.landmark[8])
                finger_tips.append(hand_landmarks.landmark[12])
                finger_tips.append(hand_landmarks.landmark[16])
                finger_tips.append(hand_landmarks.landmark[20])

        for points in finger_tips:
            avg_x += points.x
            avg_y += points.y
        
        avg_x = avg_x/5
        avg_y = avg_y/5

        for i in range(0, 5):
            dist = math.sqrt(math.pow(finger_tips[i].x-avg_x, 2) + math.pow(finger_tips[i].y-avg_y, 2));
            if dist >= 0.06:
                return False

        return True
    
    def center_of_mass(self, img, handNo): #handNo: 0 = first hand, 1 = second hand
        ret = []
        if self.results.multi_hand_landmarks:
            hands = self.results.multi_hand_landmarks[handNo]
            avg_x = 0
            avg_y = 0
            for ind, lm in enumerate(hands.landmark):
                avg_x += lm.x * img.shape[1]
                avg_y += lm.y * img.shape[0]
            avg_x /= 21
            avg_y /= 21
            ret.append([int(avg_x), int(avg_y)])
        
        return ret

    def palm_center(self, img, handNo):
        ret = []
        if self.results.multi_hand_landmarks:
            hands = self.results.multi_hand_landmarks[handNo]
            avg_x = 0
            avg_y = 0
            # 0, 5, 9, 13, 17
            for ind in range(5, 21, 4):
                avg_x += hands.landmark[ind].x * img.shape[1]
                avg_y += hands.landmark[ind].y * img.shape[0]
            avg_x += hands.landmark[0].x * img.shape[1]
            avg_y += hands.landmark[0].y * img.shape[0]
            avg_x /= 5
            avg_y /= 5
            ret.append([int(avg_x), int(avg_y)])
        
        return ret

    def dist(self, x1, y1, x2, y2):
        return math.sqrt(math.pow(x1-x2, 2) + math.pow(y1-y2, 2))

    def help(self):
        print("OpenCV x Mediapipe Hand Tracking Module made by Siwei Du")
        print("+--------------------------------------------------------------------------------------------------------------------+")
        print("Functions: ")
        print("findHands():\t\tFinds and plots the skeletal structure on image.[Returns an Image]")
        print("findPosition():\t\tFinds the position of each hand landmark.[Returns a list]")
        print("fingerCount():\t\tCounts how many fingers are held up. Works best with upright hand.[Returns a list]")
        print("checkGrab():\t\tChecks if the hand makes a grab gesture.[Returns a boolean]")
        print("center_of_mass():\tFinds the center of mass of the hand, indicated by a red dot.[Returns a list]")
        print("palm_center():\t\tFinds the center of the palm, indicated by a blue dot.[Returns a list]")
        print("dist():\t\t\tFinds the distance between two points.[Returns a float]")
        print("+--------------------------------------------------------------------------------------------------------------------+")
        

def main():
    ptime = 0
    ctime = 0

    capture = cv.imread("airport_stock_footage1.mp4")
    detector = HandDetector()

    while True:
        success, frame = capture.read()

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        detector.findHands(frame)
        landmarkList = detector.findPosition(frame)
        if len(landmarkList) != 0:
            print(landmarkList[8])

        cv.putText(
                    frame, 
                    str(int(fps)), 
                    (50, 50), 
                    cv.FONT_HERSHEY_PLAIN, 
                    3, 
                    (255, 0, 255), 
                    3
                )

        cv.imshow('Video', frame)


        if cv.waitKey(1) == 27:     # ESC to break
            break



if __name__ == "__main__":
    main()