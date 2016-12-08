# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 23:33:57 2016

@author: user
"""

import cv2


class Target:

    def __init__(self): 
        capture = cv2.VideoCapture('D://senior/CCL/video/April30/April30_2sentence1.mpg') 
        cv2.namedWindow("Target", 1)


    def run(self):
        frame = cv2.QueryFrame(self.capture)
        frame_size = cv2.GetSize(frame)
        fps=cv2.GetCaptureProperty(self.capture, cv2.CV_CAP_PROP_FPS)

        color_image = cv2.CreateImage(cv2.GetSize(frame), 8, 3)
        grey_image = cv2.CreateImage(cv2.GetSize(frame), cv2.IPL_DEPTH_8U, 1)
        moving_average = cv2.CreateImage(cv2.GetSize(frame), cv2.IPL_DEPTH_32F, 3)

        # Create Kalman Filter
        kalman = cv2.CreateKalman(4, 2, 0)
        kalman_state = cv2.CreateMat(4, 1, cv2.CV_32FC1)
        kalman_process_noise = cv2.CreateMat(4, 1, cv2.CV_32FC1)
        kalman_measurement = cv2.CreateMat(2, 1, cv2.CV_32FC1)

        first = True
        second=True
        n=0
        cp11 = []
        cp22 = []
        center_point1 = []
        predict_pt1 = []
        count=0

        while True:
            closest_to_left = cv2.GetSize(frame)[0]
            closest_to_right = cv2.GetSize(frame)[1]

            color_image = cv2.QueryFrame(self.capture)

            cv2.Smooth(color_image, color_image, cv2.CV_GAUSSIAN, 3, 0)
            if first:
                difference = cv2.CloneImage(color_image) #fully copies the image.
                temp = cv2.CloneImage(color_image)
                cv2.ConvertScale(color_image, moving_average, 1.0, 0.0) 
                first = False 
            else:          
                cv2.RunningAvg(color_image, moving_average, 0.02, None) 

            cv2.ConvertScale(moving_average, temp, 1.0, 0.0)

            # Minus the current frame from the moving average.
            cv2.AbsDiff(color_image, temp, difference) 

            # Convert the image to grayscale.
            cv2.CvtColor(difference, grey_image, cv2.CV_RGB2GRAY)

            # Convert the image to black and white.
            cv2.Threshold(grey_image, grey_image, 70, 255, cv2.CV_THRESH_BINARY)

            # Dilate and erode to get people blobs
            cv2.Dilate(grey_image, grey_image, None, 18)  
            cv2.Erode(grey_image, grey_image, None, 10) 

            storage = cv2.CreateMemStorage(0) 
            contour = cv2.FindContours(grey_image, storage, cv2.CV_RETR_CCOMP, cv2.CV_CHAIN_APPROX_SIMPLE)

            points = []

            i=0
            k=0
            while contour:
                area=cv.ContourArea(list(contour))
                #print area
                bound_rect = cv2.BoundingRect(list(contour))
                contour = contour.h_next() 
                if (area > 1500.0):
                    pt1 = (bound_rect[0], bound_rect[1])
                    pt2 = (bound_rect[0] + bound_rect[2], bound_rect[1] + bound_rect[3])
                    points.append(pt1)
                    points.append(pt2)
                    cv2.Rectangle(color_image, pt1, pt2, cv2.CV_RGB(255,0,0), 1)

                    cp1 = bound_rect[0] + (bound_rect[2]/2)
                    cp2 = bound_rect[1] + (bound_rect[3]/2)
                    cp11.append(cp1)
                    cp22.append(cp2)

                    # set previous state for prediction
                    kalman.state_pre[0,0]  = cp1
                    kalman.state_pre[1,0]  = cp2
                    kalman.state_pre[2,0]  = 0
                    kalman.state_pre[3,0]  = 0

                    # set kalman transition matrix
                    kalman.transition_matrix[0,0] = 1
                    kalman.transition_matrix[0,1] = 0
                    kalman.transition_matrix[0,2] = 0
                    kalman.transition_matrix[0,3] = 0
                    kalman.transition_matrix[1,0] = 0
                    kalman.transition_matrix[1,1] = 1
                    kalman.transition_matrix[1,2] = 0
                    kalman.transition_matrix[1,3] = 0
                    kalman.transition_matrix[2,0] = 0
                    kalman.transition_matrix[2,1] = 0
                    kalman.transition_matrix[2,2] = 0
                    kalman.transition_matrix[2,3] = 1
                    kalman.transition_matrix[3,0] = 0
                    kalman.transition_matrix[3,1] = 0
                    kalman.transition_matrix[3,2] = 0
                    kalman.transition_matrix[3,3] = 1

                    # set Kalman Filter
                    cv2.SetIdentity(kalman.measurement_matrix, cv2.RealScalar(1))
                    cv2.SetIdentity(kalman.process_noise_cov, cv2.RealScalar(1e-5))
                    cv2.SetIdentity(kalman.measurement_noise_cov, cv2.RealScalar(1e-1))
                    cv2.SetIdentity(kalman.error_cov_post, cv2.RealScalar(1))

                    #Prediction
                    kalman_prediction = cv2.KalmanPredict(kalman)
                    predict_pt  = (int(kalman_prediction[0,0]),int( kalman_prediction[1,0]))
                    predict_pt1.append(predict_pt)
                    print "Prediction",predict_pt
                    #Correction
                    kalman_estimated = cv2.KalmanCorrect(kalman, kalman_measurement)
                    state_pt = (kalman_estimated[0,0], kalman_estimated[1,0])

                    #measurement
                    kalman_measurement[0, 0] = center_point[0]
                    kalman_measurement[1, 0] = center_point[1]

            while(i<count):
                cv2.Circle(color_image, (cp11[i], cp22[i]), 1, cv2.CV_RGB(255, 100, 0), 1)


                cv2.Circle(color_image, predict_pt1[i], 1, cv2.CV_RGB(0, 255, 0), 1)
                i=i+1
            cv2.ShowImage("Target", color_image)

            c = cv2.WaitKey(int(fps))  
            if c == 27: 
                break

if __name__=="__main__":
    t = Target()
    t.run()