# OpenCV & Mediapipe Hand Tracking Module
A hand tracking module for OpenCV and Mediapipe. Contains various functions such as counting fingers, finding hands on an image, finding positions of landmarks (return a list), checking if the hand is grabbing, etc.

# Functions
```
Functions:
findHands():            Finds and plots the skeletal structure on image.                    [Returns an Image]
findPosition():         Finds the position of each hand landmark.                           [Returns a list]
fingerCount():          Counts how many fingers are held up. Works best with upright hand.  [Returns a list]
checkGrab():            Checks if the hand makes a grab gesture.                            [Returns a boolean]
center_of_mass():       Finds the center of mass of the hand, indicated by a red dot.       [Returns a list]
palm_center():          Finds the center of the palm, indicated by a blue dot.              [Returns a list]
dist():                 Finds the distance between two points.                              [Returns a float]
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
The module is not perfect, there are still small limitations such as CoM and palm center for two hands.
It will hopefully be updated regularly.

Thanks!
