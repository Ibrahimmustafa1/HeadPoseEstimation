import numpy as np
import glob
import scipy.io as sio
from math import cos, sin
from pathlib import Path
import pandas as pd
import mediapipe
import warnings     
import pandas as pd
import cv2
import pickle
warnings.filterwarnings('ignore')
faceModule = mediapipe.solutions.face_mesh

svr = pickle.load(open('SVR_model.sav', 'rb'))

def extract_landmarks(image):
    img_features = []
    img_features2 = []
    with faceModule.FaceMesh(static_image_mode=True) as faces:
        results = faces.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks != None: 
            for face in results.multi_face_landmarks:
                for landmark in face.landmark:
                    x = landmark.x
                    y = landmark.y
                    shape = image.shape 
                    img_features.append(x)
                    img_features.append(y)
                    # cv2.circle(image, (relative_x, relative_y), radius=1, color=(0, 255, 0), thickness=2)
            # the point of nose i 5
            if len(img_features) > 0:
                img_features2 = img_features.copy()
                img_features[0::2] = (img_features[0::2] - np.mean(img_features[0::2])) / max(img_features[0::2] - np.mean(img_features[0::2]))
                img_features[1::2] = (img_features[1::2] - np.mean(img_features[1::2])) / max(img_features[1::2] - np.mean(img_features[1::2]))
    return img_features , img_features2


def predict_pose(image, model):
    img_features , img_features2 = extract_landmarks(image)
    if image.shape == (0,) or len(img_features) == 0:
        return None
    img_features = np.array(img_features).reshape(1,-1)
    return model.predict(img_features),img_features2

#exort this function to the app.py
def draw_axis(img, pitch,yaw,roll, tdx=None, tdy=None, size = 100):

    yaw = -yaw
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img
def process_video():
    video_path = 'video.mp4'
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('svr.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pose , img_features  = predict_pose(frame, svr) or (None,None)
        if pose is not None:
            pitch, yaw, roll = pose[0]

            # iwant the postion of arrows to be in the middle of the face so what i shoudld give in tdx and tdy in draw axis function
            draw_axis(frame, pitch, yaw,roll,tdx=img_features[10] * frame_width,tdy=img_features[11] * frame_height ,size = 100)
        out.write(frame)

