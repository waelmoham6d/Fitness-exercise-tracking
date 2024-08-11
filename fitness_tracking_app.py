import streamlit as st
import mediapipe as mp
import numpy as np
import cv2
st.title('train with me :sunglasses:')
st.subheader("Choose the exercise we will start with 🏋️🏋️")


button_0 = st.button('Press and tracking your biceps curl train' )
button_1 = st.button('Press and trscking your squat exercise')

mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose

def calculate_angel(a,b,c):

        a=np.array(a)
        b=np.array(b)
        c=np.array(c)
        
        radianc=np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
        angle=np.abs(radianc*180.0/np.pi)

        if angle>180:
            angle=360-angle

        return angle 

if button_0:
    cap=cv2.VideoCapture(0)

    l_stage=None
    l_counter=0


    R_stage=None
    R_counter=0

       

    with mp_pose.Pose(min_detection_confidence=0.7,min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret,farme=cap.read()

            Image=cv2.cvtColor(farme,cv2.COLOR_BGR2RGB)
            Image.flags.writeable=False

            results=pose.process(Image)

            Image.flags.writeable=True
            Image=cv2.cvtColor(Image,cv2.COLOR_RGB2BGR)


            try:

                landmarks=results.pose_landmarks.landmark


                l_shoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_elbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                l_angle=calculate_angel(l_shoulder,l_elbow,l_wrist)

                R_shoulder=[landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                R_elbow=[landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                R_wrist=[landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                R_angle=calculate_angel(R_shoulder,R_elbow,R_wrist)

                cv2.putText(Image,str(int(l_angle)),
                                tuple(np.multiply(l_elbow,[640,480]).astype(int)),
                                cv2.FONT_HERSHEY_COMPLEX,0.5,(255,66,66),2,cv2.LINE_8
                                )

                cv2.putText(Image,str(int(R_angle)),
                                tuple(np.multiply(R_elbow,[640,480]).astype(int)),
                                cv2.FONT_HERSHEY_COMPLEX,0.5,(255,66,66),2,cv2.LINE_8
                                )

                if l_angle > 160:
                    l_stage = "down"
                if l_angle < 20 and l_stage =='down':
                    l_stage="up"
                    l_counter +=1

                if R_angle > 160:
                    R_stage = "down"
                if R_angle < 20 and R_stage =='down':
                    R_stage="up"
                    R_counter +=1
            except:
                pass
            # cv2.rectangle(Image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(Image, 'LEFT REPS:', (1,50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(Image, str(l_counter), 
                        (125,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(Image, 'LEFT STAGE:',(1,110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(Image, l_stage, 
                        (130,110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            #######################Right -->

            cv2.putText(Image, 'RIGHT REPS:', (400,50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(Image, str(R_counter), 
                        (500,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(Image, 'RIGHT STAGE:',(400,110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(Image, R_stage, 
                        (510,110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)


            mp_drawing.draw_landmarks(Image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)                               
                                )        
            
            cv2.imshow('Mediapipe Feed',Image)


            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if button_1:

    stage=None
    counter=0

    with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
        cap=cv2.VideoCapture(0)
        while cap.isOpened():  
            ret,farme=cap.read()

            Image=cv2.cvtColor(farme,cv2.COLOR_BGR2RGB)

            Image.flags.writeable=False

            results=pose.process(Image)

            Image.flags.writeable=True
            Image=cv2.cvtColor(Image,cv2.COLOR_RGB2BGR)


            try:

                landmarks=results.pose_landmarks.landmark


                l_hip=[landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                l_knee=[landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                l_ankle=[landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                l_angle=calculate_angel(l_hip,l_knee,l_ankle)


                R_hip=[landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                R_knee=[landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                R_ankle=[landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                R_angle=calculate_angel(R_hip,R_knee,R_ankle)


                cv2.putText(Image,str(int(l_angle)),
                                tuple(np.multiply(l_knee,[640,480]).astype(int)),
                                cv2.FONT_HERSHEY_COMPLEX,0.5,(255,66,66),2,cv2.LINE_8
                                )

                cv2.putText(Image,str(int(R_angle)),
                                tuple(np.multiply(R_knee,[640,480]).astype(int)),
                                cv2.FONT_HERSHEY_COMPLEX,0.5,(255,66,66),2,cv2.LINE_8
                                )
                
                if l_angle > 150 or R_angle> 150 :
                    stage = "stand"
                if (l_angle < 100 or R_angle<100) and (stage =='stand'):
                    stage="down"
                    counter +=1

               
                
            except:
                pass

            cv2.putText(Image,'REPS:', (1,50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(Image,str(counter), 
                        (125,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(Image,'STAGE:',(1,110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(Image,stage, 
                        (130,110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            mp_drawing.draw_landmarks(Image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)                               
                                    )        
                
            cv2.imshow('Mediapipe Feed',Image)


            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()        



























































