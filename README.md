# OpenCV & Mediapipe Hand Tracking Module
A hand tracking module for OpenCV and Mediapipe. Contains various functions such as counting fingers, finding hands on an image, finding positions of landmarks (return a list), checking if the hand is grabbing, etc.

# Functions
```
Functions:
findHands():            Finds and plots the skeletal structure on image.                    [Returns an Image]
findPosition():         Finds the position of each hand landmark.                           [Returns a list]
fingerCount():          Counts how many fingers are held up. Works best with upright hand.  [Returns a list]
checkGrab():            Checks if the hand makes a grab gesture.                            [Returns a boolean]
checkGrabAlt():         Alternative method for checkGrab(). [Returns a boolean]
swipeDirection():       Checks which way the user swipes their hand while grabbing. [Returns a string]
center_of_mass():       Finds the center of mass of the hand, indicated by a red dot.       [Returns a list]
palm_center():          Finds the center of the palm, indicated by a blue dot.              [Returns a list]
dist():                 Finds the distance between two points.                              [Returns a float]
```

# How to use:
* create detector object and then call ```.findHands() and .findPosition()``` at beginning of each OpenCV iteration.
* Call the functions as needed, they can be called in the loop.
* Example:

```python
# Declare stuff

detector = hm.HandDetector(detectionCon = 0.75, trackCon = 0.6)

while cap.isOpened():

    success, frame = cap.read()

    # You can change the display window size by resizing frame here.

    img = detector.findHands(frame) # Draws hands onto img

    detector.findPosition(frame, 2)

    # do stuff:
    # If you want to draw stuff or add text, draw it onto img, not frame.


    cv.imshow('Video', img)         # Display img.

cap.release()
cv.destroyAllWindows()
```

# Libraries:
```
opencv
mediapipe
numpy
math
time
```

# Other things
The module is not perfect, support for multiple hands in the frame at the same time is still being developed.
It will hopefully be updated regularly.

Thanks!
