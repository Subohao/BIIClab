import numpy as np
import cv2

cap = cv2.VideoCapture('D://senior/CCL/video/April30/April30_2sentence1.mpg')

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
# r,h,c,w - region of image
#           simply hardcoded the values
r,h,c,w = 130,250,180,75  
track_window = (c,r,w,h)

r2,h2,c2,w2 = 130,270,420,50
track_window2 = (c2,r2,w2,h2)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
roi2 = frame[r2:r2+h2,c2:c2+w2]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#threshold the HSV image to get certain color
mask = cv2.inRange(hsv_roi, np.array((0., 0.,0.)), np.array((50.,50.,30.)))
#for black (0, 0, 0) & (29, 30, 30)
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    #bool ret if is 1, read correctly
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        ret, track_window2 = cv2.meanShift(dst, track_window2, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        x2,y2,w2,h2 = track_window2
        cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(0,0,255),2)
        cv2.imshow('img2',frame)
#        cv2.imshow(dst)

        k = cv2.waitKey(30) & 0xff
        if k == ord("q"):
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()