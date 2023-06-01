# -*- coding = utf-8 -*-
# @Time :2023/4/19 17:44
# @Author : AgentLee
# @File : holistic.py
# @Software : PyCharm
import copy
import argparse

import cv2 as cv
import numpy as np
import math
import mediapipe as mp
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Z_PEAKS=[]
shoulder_axisZ = []
Rheel_axisZ = [] #  全部
shoulder_list = [] # 峰值
fps = 30
m = 67
v_linear = []  #用于存放所有点位速度
flag = 0
standard = 0
def test():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(111, projection="3d")
    ax = plt.axes(projection='3d')
    fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)
    cap = cv.VideoCapture('./src/cxj.mp4')
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
        # 不关闭窗口
        cap.release()
        plt.close()
        cv.destroyAllWindows()
def plot_world_landmarks(
        plt,
        ax,
        landmarks,
        visibility_th=0.5,
):
    landmark_point = []
    global standard
    global flag  # 记录标准锁
    for index, landmark in enumerate(landmarks.landmark):
        if not flag and index == 30:
            standard = [landmark.visibility, (
            float(format(landmark.x, '.5f')), float(format(landmark.y, '.5f')), float(format(landmark.z, '.5f')))]
            flag = 1
            print(standard)
    for index, landmark in enumerate(landmarks.landmark):
        landmark_point.append(
            [landmark.visibility, (float(format(landmark.x, '.5f')), float(format(landmark.y, '.5f')), float(format(landmark.z, '.5f')))])
    face_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    right_arm_index_list = [11, 13, 15, 17, 19, 21]
    left_arm_index_list = [12, 14, 16, 18, 20, 22]
    right_body_side_index_list = [11, 23, 25, 27, 29, 31]
    left_body_side_index_list = [12, 24, 26, 28, 30, 32]
    shoulder_index_list = [11, 12]
    waist_index_list = [23, 24]
    # 重心模拟
    center_of_mass = [11, 12, 25, 26]

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
    Rheel_axisZ.append(landmark_point[30][1][1]*(-1))
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
    # ax.scatter(0, 0, 0)
    # ax.scatter(COM_x, COM_y, 0)
    ax.plot(right_arm_x, right_arm_y, right_arm_z)
    ax.plot(left_arm_x, left_arm_y, left_arm_z)
    ax.plot(right_body_side_x, right_body_side_y, right_body_side_z)
    ax.plot(left_body_side_x, left_body_side_y, left_body_side_z)
    ax.plot(shoulder_x, shoulder_y, shoulder_z)
    ax.plot(waist_x, waist_y, waist_z)
    # ax.scatter(landmark_point[26][1][0], landmark_point[26][1][2], landmark_point[26][1][1]*(-1))
    plt.pause(.001)

    return

def axisFilter(axis):
    peaks_and_valleys = [] # 波峰波谷（脚后跟）
    fps_num = [] # 帧数
    cal_V_list = [] # 波峰波谷＋相邻坐标
    cal_V_list_shoulder = [] # 右肩波峰波谷＋相邻坐标
    for index,item in enumerate(axis):
        if index == 0 or index == len(axis) - 1:
            continue
        if (item > axis[index-1] and item > axis[index+1]) or (item < axis[index-1] and item < axis[index+1]):
            peaks_and_valleys.append(item)
            shoulder_list.append(shoulder_axisZ[index])
            fps_num.append(index)
            cal_V_list.append([axis[index-1], item, axis[index+1]])
            cal_V_list_shoulder.append([shoulder_axisZ[index-1], shoulder_axisZ[index], shoulder_axisZ[index+1]])
    # print(cal_V_list)#peaks_and_valleys, cal_V_list)
    v_res = wattCalculate(cal_V_list, peaks_and_valleys)
    # v_res = wattCalculate(cal_V_list_shoulder, shoulder_list)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.set_title('Right Shoulder/heel ZAxis')
    ax1.plot(fps_num, peaks_and_valleys, label="Rheel_ZAxis")
    ax1.plot(fps_num, shoulder_list, label="Rshoulder_ZAxis")
    ax1.set_xlabel('Fps')
    ax1.set_ylabel('Z_Axis')
    ax1.legend(loc='best')
    ax2.plot(fps_num, v_res, label="v")
    ax2.set_xlabel('Fps')
    ax2.set_ylabel('V(m/s)')
    ax3 = ax2.twinx()
    ax3.plot(fps_num, peaks_and_valleys, color='green', label='Z_Axis')
    # ax3.set_ylabel("v_heel")
    ax2.legend(loc='upper left')
    ax3.legend(loc='upper right')
    plt.show()
def wattCalculate(v_list, axis_list):
    trans_H = 0
    trans_Vf = 0
    v_res = []
    for index, item in enumerate(v_list):
        v = abs(abs(item[2] - item[1])-abs(item[1] - item[0])) / (2 / fps)  #  7.05待定
        # print(v)
        v_res.append(v)
        # if index != len(v_list) - 1:
        #     trans_Vf += abs(math.pow(item[1], 2) - math.pow(v_list[index+1][1], 2))
    for i, v in enumerate(v_res):
        if i != len(v_res) - 1:
            trans_Vf += abs(math.pow(v, 2) - math.pow(v_res[i+1], 2))
    for i, axis in enumerate(axis_list):
        if i != len(axis_list) - 1:
            trans_H += abs(axis - axis_list[i+1])
    # 势能
    Ep = m * 9.8 * trans_H  # 7.05待定
    # 动能
    Ek = m * trans_Vf / 2
    # 总
    E = Ep + Ek
    # print(Ep, Ek, E, trans_H, trans_Vf, v_res)
    return v_res
    # W = 9.8 * sum(axis_list) + m * sum(np.square(v_linear)) / 2
test()
axisFilter(Rheel_axisZ)
