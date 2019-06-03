#this is motion detector module using open cv
# shows the time of the object appeared in front of the camera

import cv2, pandas
from datetime import datetime

first_frame = None
status_list = [None,None]
times = []
x=0
df = pandas.DataFrame(columns=["Start", "End"]) #Dataframe to store the time values during which object detection and movement appears.


video = cv2.VideoCapture(cv2.CAP_DSHOW)#VideoCapture object is used to record video using web cam

while True:
    #check is a bool data type, returns true if VideoCapture object is read
    check,frame = video.read()
    status = 0
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  # For converting the frame color to gray scale
    gray = cv2.GaussianBlur(gray,(21,21),0)  # For converting the gray scale frame to GaussianBlur

    if first_frame is None:
        first_frame = gray   # used to store the first image/frame of the video
        continue
    delta_frame = cv2.absdiff(first_frame, gray)#calculates the difference between first and other frames
    thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta, None, iterations=0) #Provides threshold value, so if the difference is <30 it will turn to black otherwise if >30 pixels will turn to white
    cnts, hierarchy = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Removing noises and shadows, any part which is greater than 1000 pixels will be converted to white
    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        status = 1 #change in status when the object is detected
        #Creating a rectangular box around the object in frame
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    #list of status for every frame
    status_list.append(status)

    status_list=status_list[-2:]

    #Record datetime in a list when change occurs
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())

    print("times",times)
    cv2.imshow('frame',frame) 
    cv2.imshow('Capturing',gray)
    cv2.imshow('delta',delta_frame)
    cv2.imshow('thresh',thresh_delta)
    key = cv2.waitKey(1) #changing the frame after 1 millisecond
    #Used for terminating the loop once 'q' is pressed
    if key==ord('q'):
        break

print(status_list)
print(times)
x=len(times)-1

for i in range(0,x):

    df = df.append({"Start":times[i], "End":times[i+1]},ignore_index=True)

df.to_csv('Times.csv')
video.release()
cv2.destroyAllWindows #will be closing all the windows