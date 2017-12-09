#Copyright (c) 2017 N Dalal
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

from collections import deque
import numpy as np
import cv2

################# INITIALIZATION #################
pts = deque(maxlen=20) #Queue size for tracking
bufferSize = 20

fps = 30 #fps processing speed for VideoWriter()

#define green mask color in HSV
#How to find HSV values to track?
#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
lower = {'green':(25, 100, 100)} #lower bound
upper = {'green':(45, 255, 255)} #upper bound

camera = cv2.VideoCapture(0)
grabbed, new_frame = camera.read() #start camera and get a frame
h, w = new_frame.shape[:2]

ratio = 0.75 #set new ratio
h=int(h*ratio) #global Height and Width
w=int(w*ratio) #global Height and Width

current_frame = cv2.resize(new_frame, (w, h))
previous_frame = current_frame

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) #set kernel for morphologyEx
maskKernel = np.ones((3,3),np.uint8) #set kernel for the mask

#set up the video for recording
fourcc = cv2.VideoWriter_fourcc(*'XVID')
ranName = np.random.randint(1000000, size=1)
video_out = cv2.VideoWriter('output_'+str(ranName[0])+'.avi',fourcc, int(fps), (w,h))

while(True):
    if not grabbed: #check that a frame is grabbed
        break

    output = current_frame.copy()
    cv2.putText(output ,'FPS: ' + str(fps), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,255,0),2)

################# MOTION DETECTION #################
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    frame_diff = cv2.absdiff(current_frame_gray,previous_frame_gray) #frame difference
    frame_diff = cv2.morphologyEx(frame_diff, cv2.MORPH_OPEN, kernel) #filter noises

    _ ,thresh1 = cv2.threshold(frame_diff,50,255,cv2.THRESH_BINARY) #thresholding of frame_diff

    nzCount = np.count_nonzero(thresh1) #count the nonzero pixels

    cv2.imshow('A Difference of two frames',frame_diff) #show the frame difference
    cv2.imshow('A Difference of two frames (Threshold = 50)',thresh1) #show the frame difference with Threshold = 50
    print(nzCount) #non black counter

    if nzCount > 1000: #check for motion
        cv2.putText(output ,'Motion', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,255,0),2) #write Motion in the frame

################# HSV CONVERTION AND MASKING #################
        blurred = cv2.GaussianBlur(output, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        #cv2.imshow('HSV',hsv) #show hsv frame

        for value in upper.items(): #checks what value or values we want to filter. In our case, we want to filter one color only
            mask = cv2.inRange(hsv, lower['green'], upper['green'])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, maskKernel) #noise reduction
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, maskKernel) #noise reduction

################# FIND CONTOURS #################
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2] #find contours

            #If you want to see the contours:
            #for c in cnts:
            #    peri = cv2.arcLength(c, True)
            #    approx = cv2.approxPolyDP(c, 0.01 , False)
            #    cv2.drawContours(output, [approx], -1, (255, 250, 0), 3)

            if len(cnts) > 0: #Proceed if there any contours

################# COMPUTE AREA AND RADIUS #################
                c = max(cnts, key=cv2.contourArea) #find the biggest contour
                area = cv2.contourArea(c) #calculate the area
                ((x, y), radius) = cv2.minEnclosingCircle(c) #find the center (x,y) and the radius of the min.Enc.circle

################## DRAW A CIRCLE #################
                if area > 1000 and radius > 10:
                    cv2.circle(output, (int(x), int(y)), int(radius), (0,0,255), 10) #draw circle
                    cv2.putText(output ,'Object Detected', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,0,255),2) #add text
                    pts.appendleft((int(x), int(y))) #store the position in a queue

################# DRAW AND UPDATE PATH #################
    queueSize = len(pts)
    queueColor = [0,0,255]
    for i in np.arange(1, queueSize):
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(bufferSize / float(i + 1)) * 2.5) #change the thikness of the tracking path
        cv2.line(output, pts[i - 1], pts[i], queueColor, thickness) #draw the line
        queueColor = [queueColor[0]+25, queueColor[1]+25, 255] #modify the color of the tracking path

################# SAVE THE FRAME AND READ A NEW ONE #################
    cv2.imshow("Output",output)
    video_out.write(output)

    previous_frame = current_frame.copy()
    ret, new_frame = camera.read()
    current_frame = cv2.resize(new_frame, (w,h))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
