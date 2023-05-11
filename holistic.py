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
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

Z_PEAKS=[]
shoulder_axisZ = []
def test():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(111, projection="3d")
    ax = plt.axes(projection='3d')
    fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)
    cap = cv.VideoCapture('./src/squat.mp4')
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
            results = holistic.process(image)  # 能返回坐标
            # results = holistic.process(cv.cvtColor(image, cv.COLOR_BGR2RGB)) # 能返回坐标
            plot_world_landmarks(
                plt,
                ax,
                results.pose_world_landmarks,
            )
            # 画图
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
            # cv.imshow('MediaPipe Holistic', image)
            if cv.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv.destroyAllWindows()
def plot_world_landmarks(
        plt,
        ax,
        landmarks,
        visibility_th=0.5,
):
    landmark_point = []
    for index, landmark in enumerate(landmarks.landmark):
        landmark_point.append(
            [landmark.visibility, (landmark.x, landmark.y, landmark.z)])
    face_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    right_arm_index_list = [11, 13, 15, 17, 19, 21]
    left_arm_index_list = [12, 14, 16, 18, 20, 22]
    right_body_side_index_list = [11, 23, 25, 27, 29, 31]
    left_body_side_index_list = [12, 24, 26, 28, 30, 32]
    shoulder_index_list = [11, 12]
    waist_index_list = [23, 24]
    # 重心模拟
    center_of_mass = [11, 12, 25, 26]
    knee = [26]

    # 面部
    face_x, face_y, face_z = [], [], []
    for index in face_index_list:
        point = landmark_point[index][1]
        face_x.append(point[0])
        # face_y.append(point[2])
        # face_z.append(point[1] * (-1))
        face_y.append(point[2])
        face_z.append(point[1] * (-1))

    # 右臂
    right_arm_x, right_arm_y, right_arm_z = [], [], []
    for index in right_arm_index_list:
        point = landmark_point[index][1]
        right_arm_x.append(point[0])
        right_arm_y.append(point[2])
        right_arm_z.append(point[1] * (-1))

    # 左臂
    left_arm_x, left_arm_y, left_arm_z = [], [], []
    for index in left_arm_index_list:
        point = landmark_point[index][1]
        left_arm_x.append(point[0])
        left_arm_y.append(point[2])
        left_arm_z.append(point[1] * (-1))

    # 右半身
    right_body_side_x, right_body_side_y, right_body_side_z = [], [], []
    for index in right_body_side_index_list:
        point = landmark_point[index][1]
        right_body_side_x.append(point[0])
        right_body_side_y.append(point[2])
        right_body_side_z.append(point[1] * (-1))

    # 左半身
    left_body_side_x, left_body_side_y, left_body_side_z = [], [], []
    for index in left_body_side_index_list:
        point = landmark_point[index][1]
        left_body_side_x.append(point[0])
        left_body_side_y.append(point[2])
        left_body_side_z.append(point[1] * (-1))

    # 肩
    shoulder_x, shoulder_y, shoulder_z = [], [], []
    for index in shoulder_index_list:
        point = landmark_point[index][1]
        shoulder_x.append(point[0])
        shoulder_y.append(point[2])
        shoulder_z.append(point[1] * (-1))

    # 腰
    waist_x, waist_y, waist_z = [], [], []
    for index in waist_index_list:
        point = landmark_point[index][1]
        waist_x.append(point[0])
        waist_y.append(point[2])
        waist_z.append(point[1] * (-1))

    # 重心
    Cal_COM_x, Cal_COM_y, Cal_COM_z = [], [], []
    for index in center_of_mass:
        point = landmark_point[index][1]
        Cal_COM_x.append(point[0])
        Cal_COM_y.append(point[2])
        Cal_COM_z.append(point[1]*(-1))
    COM_x = sum(Cal_COM_x) / 4
    COM_y = sum(Cal_COM_y) / 4
    # COM_z = sum(Cal_COM_z) / 4

    shoulder_axisZ.append(shoulder_z[0])
    ax.cla()
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.invert_yaxis()
    # Mediapipe坐标和matplotlib y z轴相反，且z(mat)需×-1
    ax.scatter(face_x, face_y, face_z)
    ax.scatter(0, 0, 0)
    ax.scatter(COM_x, COM_y, 0)
    ax.plot(right_arm_x, right_arm_y, right_arm_z)
    ax.plot(left_arm_x, left_arm_y, left_arm_z)
    ax.plot(right_body_side_x, right_body_side_y, right_body_side_z)
    ax.plot(left_body_side_x, left_body_side_y, left_body_side_z)
    ax.plot(shoulder_x, shoulder_y, shoulder_z)
    ax.plot(waist_x, waist_y, waist_z)
    ax.scatter(landmark_point[26][1][0], landmark_point[26][1][2], landmark_point[26][1][1]*(-1))
    plt.pause(.001)

    return

# def axisFilter(axis):

test()