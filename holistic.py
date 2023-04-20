# -*- coding = utf-8 -*-
# @Time :2023/4/19 17:44
# @Author : AgentLee
# @File : holistic.py
# @Software : PyCharm
import copy
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp
from vpython import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

points = []

c = []
for x in range(33):
    points.append(sphere(radius=5, pos=vector(0, -50, 0)))
    c.append(curve(retain=2, radius=4))
cap = cv.VideoCapture('./src/run.mp4')
with mp_holistic.Holistic(
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # 加载一个视频的话，把continue换成break
      break

    image.flags.writeable = False
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = holistic.process(image) # 能返回坐标
    #画图
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())

    # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    cv.imshow('MediaPipe Holistic', cv.flip(image, 1))
    if cv.waitKey(5) & 0xFF == 27:
        break

cap.release()