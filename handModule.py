#from turtle import Turtle
import mediapipe as mp
import cv2 as cv
import numpy as np
import math
import time


class HandDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon= 0.5, trackCon = 0.5, grabbed = False, grabPos = []):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Mediapipe hands solution, might need to be updated 
        # as Mediapipe gets updated in the future
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
                                        self.mode,
                                        self.maxHands,
                                        1,
                                        self.detectionCon,
                                        self.trackCon
                                        )
        self.mp_draw = mp.solutions.drawing_utils

        # Variables for swipe direction function
        self.grabbed = grabbed
        self.grabPos = grabPos
        self.direction = "None"

    def findHands(self, img, draw = True):
        # Make img RGB so it can be used in the process function
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

        return img
    
    def findPosition(self, img, handNo = 0, draw = True):
        """findPosition()

        Args:
            img:    an img to process and find hands on
            handNo: number of hands on the screen
            draw:   Used for debugging purposes
        Returns:
            list: a list of coordinates representing the points on a hand
                  according to this image:
                  https://google.github.io/mediapipe/images/mobile/hand_landmarks.png    
        """
        
        self.retList = []

        # Can be improved...
        if handNo == 2:
            if self.results.multi_hand_landmarks:
                for hands in self.results.multi_hand_landmarks:
                    for ind, lm in enumerate(hands.landmark):
                        h, w, c = img.shape
                        # h, w = 1, 1
                        cx, cy = int(w*lm.x), int(h*lm.y)
                        self.retList.append([ind, cx, cy])
        else:
            if self.results.multi_hand_landmarks:
                currentHand = self.results.multi_hand_landmarks[handNo]
                for ind, lm in enumerate(currentHand.landmark):
                    h, w, c = img.shape
                    # h, w = 1, 1
                    cx, cy = int(w*lm.x), int(h*lm.y)
                    self.retList.append([ind, cx, cy])

        return self.retList

    def fingerCount(self, img, handNo = 0, draw = True):
        """fingerCount()

        Args:
            img:    an image to process and find the hands on.
            handNo: number of hands on screen.
            draw:   debugging purposes.
        Returns:
            list: a list of numbers representing each finger
        
        Other Variables:
            SHIFT: which joint is used in comparison against the fingertips
                   when running the algorithm.
                   The algorithm works by determining whether the fingertip to palm
                   distance is shorter than the specified joint to palm distance.
                   In this case, we are using the 1st joint from the knuckles with
                   SHIFT = 2
        """

        # thumb: 4 = tip
        # index: 8 = tip, 5 = knuckle
        # middle: 12 = tip, 9 = knuckle
        total = 0
        indexList = []

        SHIFT = 2;

        # Making sure there are hands within the camera frame first.
        if self.results.multi_hand_landmarks:
            # Palm and center of mass coordinates.
            # center = self.center_of_mass(img)    
            palm_c = self.palm_center(img)

            # Calculating the distances for index to pinky.
            for i in range(8, 24, 4):
                distA = self.dist(palm_c[0][0], palm_c[0][1], self.retList[i][1], self.retList[i][2])
                distB = self.dist(palm_c[0][0], palm_c[0][1], self.retList[i-SHIFT][1], self.retList[i-SHIFT][2])
                # cv.circle(img, (self.retList[i][1], self.retList[i][2]), 10, (255, 255, 255), 10)
                
                # print(i," ",distA, " " ,distB)
                if distA > distB:
                    indexList.append(i)
                    total+=1

            # Calculating thumb, separate from others due to physical differences.
            # So, thumb might need to be treated differently in the future.
            distA = self.dist(palm_c[0][0], palm_c[0][1], self.retList[4][1], self.retList[4][2])
            distB = self.dist(palm_c[0][0], palm_c[0][1], self.retList[4-SHIFT][1], self.retList[4-SHIFT][2])
            
            if (distA > distB):
                indexList.append(4)
                total+=1

        return indexList

    def checkGrab(self, img, draw = True):
        """checkGrab()

        This function works by calculating the avg point of the finger tips
        and seeing if the distance from each fingertip is within a certain distance
        to that average point.

        Args:
            img:    an image to process
            handNo: number of hands in frame
            draw:   Debugging purposes.
        Returns:
            Boolean: if the hand in frame is making a grabbing gesture.    
        """


        avg_x = 0
        avg_y = 0

        finger_tips = []

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                finger_tips = hand_landmarks.landmark[4:21:4]

            for points in finger_tips:
                avg_x += points.x
                avg_y += points.y
            
            avg_x = avg_x/5
            avg_y = avg_y/5

            for i in range(0, 5):
                dist = math.sqrt(math.pow(finger_tips[i].x-avg_x, 2) + math.pow(finger_tips[i].y-avg_y, 2));
                if dist <= 0.02:
                    return True

        return False
    
    def checkGrabAlt(self, img, handNo = 0, draw = True, debug = False):
        """checkGrabAlt()

        This function is an alternative way of checking for grabbing gesture.
        It is much simpler and easier to understand compared to the original.
        But I will keep the original because I am too lazy to delete it and it 
        might become helpful in the future.
        
        This function works by checking the distance between the center of the palm
        and the center of mass of the hand to see if they are within a certain
        distance of each other.

        Args:
            img:    an image to process
            handNo: number of hands in frame
            draw:   debugging purposes
            debug:  even more debugging purposes (I need to clean this up soon)
        Returns:
            Boolean: if the hand is making a grabbing gesture.
        """

        if self.results.multi_hand_landmarks:
            mass = self.center_of_mass(img)
            cent = self.palm_center(img)

            if debug:
                print("+-----CheckGrabAlt DEBUG-----+")
                print("mass: ", mass)
                print("cent: ", cent)
                print("dist: ", self.dist(mass[0][0], mass[0][1], cent[0][0], cent[0][1]))


            if self.dist(mass[0][0], mass[0][1], cent[0][0], cent[0][1]) < 100:
                return True

        return False

    def checkGrabCnt(self, img, draw = True, debug = False):
        grabCnt = 0
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                avg_x, avg_y = 0, 0
                fingerTips = hand_landmarks.landmark[4:21:4]
                print(fingerTips[0].x)
                for points in fingerTips:
                    avg_x += points.x
                    avg_y += points.y
                
                avg_x = avg_x/5
                avg_y = avg_y/5

                for i in range(0, 5):
                    dist = math.sqrt(math.pow(fingerTips[i].x-avg_x, 2) + math.pow(fingerTips[i].y-avg_y, 2))
                    if dist <= 0.025:
                        grabCnt += 1
                        break
        else:
            print("[Error! No hands detected]")

        return grabCnt

    def swipeDirection(self, img, debug = False):
        """swipeDirection()

        NOTE: ONLY WORKS AS INTENDED WHEN 1 HAND IS DETECTED

        Concept: To determine the direction swiped, we use the 
        release position relative to the initial grab position
        
        We can set the initial grab position as the origin of a 
        cartesian plane, and split the plane into 4 regions with 2
        lines that form an 'X'. Which ever region the release position
        lands in is the direction being swiped to.

        The 'X' will be formed by equations: y = +-(height/width)x
        we just need to check whether the finished y position is above 
        or below each of the two lines. From this we can easily determine 
        which region the hand ended up in

        Args:
            img:    an image to process on
            debug:  debugging purposes.
        Returns:
            string: the direction of the grab-swipping gesture

        """
        ret = "None"
        
        if not self.results.multi_hand_landmarks:
            return ret

        # This is a toggle structure.
        if self.checkGrab(img):
            if self.grabbed == False:
                self.grabbed = True
                self.grabPos = self.center_of_mass(img)
        else:
            if self.grabbed == True:
                self.grabbed = False
                
                # Calculate direction:
                # Getting final position of hand when releasing grab
                finalPos = self.center_of_mass(img)
                
                grab_x = self.grabPos[0][0]
                grab_y = self.grabPos[0][1]
                fin_x = finalPos[0][0]
                fin_y = finalPos[0][1]

                width = img.shape[1]
                height = img.shape[0]

                # Set initial position as origin:
                fin_x -= grab_x
                fin_y -= grab_y

                # Variables for determine which region, 
                # (a, b): (1, 1) top, (1, 0) left, (0, 1) right, (0, 0) bottom
                a = -1
                b = -1

                if fin_y >= (height/width) * fin_x:
                    a = 1;
                else:
                    a = 0
                if fin_y >= -(height/width) * fin_x:
                    b = 1;
                else:
                    b = 0

                # Camera is mirror, so go opposite
                if a == 1:
                    if b == 1:
                        ret = "Down"
                    else:
                        ret = "Right"                
                else:
                    if b == 1:
                        ret = "Left"
                    else:
                        ret = "Up"

        return ret

    def center_of_mass(self, img): #handNo: 0 = first hand, 1 = second hand
        """center_of_mass()
        
        Args:
            img:    An image to process on
        Returns:
            list: coordinates of the center of mass of the hand(s) based on
                  the landmarks.

        """
        
        ret = []
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                avg_x = 0
                avg_y = 0
                for ind, lm in enumerate(hand_landmarks.landmark):
                    avg_x += lm.x * img.shape[1]
                    avg_y += lm.y * img.shape[0]
                avg_x /= 21
                avg_y /= 21
                ret.append([int(avg_x), int(avg_y)])
        
        return ret

    def palm_center(self, img):
        """palm_center()
        
        Args:
            img:    An image to process on
        Returns:
            list: coordinates of the center of palm(s)

        """
        ret = []
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                ## 0, 5, 9, 13, 17, points on the palm triangle on MediaPipe documentation.
                fingerTips = hand_landmarks.landmark[5:18:4]
                avg_x = 0
                avg_y = 0
            
                for coords in fingerTips:
                    avg_x += coords.x * img.shape[1]
                    avg_y += coords.y * img.shape[0]

                # 2 Times for base of hand for better accuracy
                avg_x += 2 * hand_landmarks.landmark[0].x * img.shape[1]
                avg_y += 2 * hand_landmarks.landmark[0].y * img.shape[0]

                avg_x /= 6
                avg_y /= 6
                ret.append([int(avg_x), int(avg_y)])
        
        return ret

    def dist(self, x1, y1, x2, y2):
        """dist()

        Args:
            x1: x position of first point
            y1: y position of first point
            x2: x position of second point
            y2: y position of second point
        Returns:
            float: a number in pixels representing the distance.
        """
        return math.sqrt(math.pow(x1-x2, 2) + math.pow(y1-y2, 2))

    def help(self):
        """help()
        
        A helper function designed to print out all available features of
        the module as well as some helpful tips with using the module.

        """
        print("\n\n\nOpenCV x Mediapipe Hand Tracking Module made by Siwei Du")
        print("+--------------------------------------------------------------------------------------------------------------------+")
        print("Functions: ")
        print("findHands():\t\tFinds and plots the skeletal structure on image.[Returns an Image]")
        print("findPosition():\t\tFinds the position of each hand landmark.[Returns a list]")
        print("fingerCount():\t\tCounts how many fingers are held up. Works best with upright hand.[Returns a list]")
        print("checkGrab():\t\tChecks if the hand makes a grab gesture.[Returns a boolean]")
        print("checkGrabAlt():\t\tAlternative method for checkGrab(). [Returns a boolean]")
        print("swipeDirection():\t\tChecks which way the user swipes their hand while grabbing. [Returns a string]")
        print("center_of_mass():\tFinds the center of mass of the hand, indicated by a red dot.[Returns a list]")
        print("palm_center():\t\tFinds the center of the palm, indicated by a blue dot.[Returns a list]")
        print("dist():\t\t\tFinds the distance between two points.[Returns a float]")
        print("+--------------------------------------------------------------------------------------------------------------------+\n>>\n")
        print("!!!!!!!!!!\tREAD FOR CLARIFICATION\t!!!!!!!!!!")
        print("* For maximum accuracy, make sure environment is: ")
        print("*** Well lit, palm facing camera, single hand in frame.\n\n")
        print("* Swipe Direction:")
        print("*** To use, grab hand, move hand, release hand.")
        print("*** Grab gesture works best with fist, or pinching all your fingers together")
        print("*** so that the fingertips are as close as possible.")
        print("*** Works best if you exaggerate the grab and release gestures\n\n")


# ----------------------------- Main 
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
