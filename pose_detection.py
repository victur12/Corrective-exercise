from operator import delitem
from turtle import color
import cv2
import mediapipe as mp
import csv
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("push_up.mp4");
class_name = "push_up"

with mp_pose.Pose(
    static_image_mode = False) as pose:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks is not None:
            mp_drawing.draw_landmarks (frame, results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), circle_radius =3),
               )
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

try:
            # Extract Pose landmarks
    pose = results.pose_landmarks.landmark
    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose]).flatten())
    row = pose_row
    
    # Append class name 
    row.insert(0, class_name)
    
    # Export to CSV
    with open('coords.csv', mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(row) 
except:
    pass


print('termino')
cap.release()
cv2.destroyAllWindows()
            