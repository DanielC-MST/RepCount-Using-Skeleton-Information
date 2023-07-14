# _*_ coding: utf-8 _*_
# @Author: Haodong_Chen
# @Time: 7/12/23 1:47 PM
import pandas as pd
import numpy as np
import os
import cv2
from mediapipe.python.solutions import pose as mp_pose
import torch.onnx
import time
import yaml
import argparse


torch.multiprocessing.set_sharing_strategy('file_system')


def calculate_average_angle(left_angles, right_angles):
    average_angles = (left_angles + right_angles) / 2
    return average_angles


def normalize_angles_horizontal(angles_landmarks):
    max_value = np.max(angles_landmarks)
    min_value = np.min(angles_landmarks)

    normalized_angle_landmarks = (angles_landmarks - min_value) / (max_value - min_value)

    return normalized_angle_landmarks


# Calculate the joint angles between two pose points
def calculate_angle(joint1, joint2, joint3):
    # Calculate the vectors for the joints
    vector1 = joint1 - joint2
    vector2 = joint3 - joint2

    # Calculate the dot product and the magnitudes of the vectors
    dot_product = np.dot(vector1, vector2)
    magnitudes = np.linalg.norm(vector1) * np.linalg.norm(vector2)

    # Calculate the cosine of the angle
    cosine_angle = dot_product / magnitudes

    # Calculate the angle in radians and convert it to degrees
    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def normalize_landmarks(all_landmarks):
    x_max = np.expand_dims(np.max(all_landmarks[:, :, 0], axis=1), 1)
    x_min = np.expand_dims(np.min(all_landmarks[:, :, 0], axis=1), 1)

    y_max = np.expand_dims(np.max(all_landmarks[:, :, 1], axis=1), 1)
    y_min = np.expand_dims(np.min(all_landmarks[:, :, 1], axis=1), 1)

    z_max = np.expand_dims(np.max(all_landmarks[:, :, 2], axis=1), 1)
    z_min = np.expand_dims(np.min(all_landmarks[:, :, 2], axis=1), 1)

    all_landmarks[:, :, 0] = (all_landmarks[:, :, 0] - x_min) / (x_max - x_min)
    all_landmarks[:, :, 1] = (all_landmarks[:, :, 1] - y_min) / (y_max - y_min)
    all_landmarks[:, :, 2] = (all_landmarks[:, :, 2] - z_min) / (z_max - z_min)

    all_landmarks = all_landmarks.reshape(len(all_landmarks), -1)
    return all_landmarks


def main(args):
    pre_time = time.time()
    if os.path.isfile(args.config):
        with open(args.config, "r") as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)
    else:
        raise ValueError("Config file does not exist.")

    root_dir = config['dataset']['dataset_root_dir']
    # test poses
    test_pose_save_dir = os.path.join(root_dir, args.output)
    test_video_dir = os.path.join(root_dir, 'video/test')
    label_dir = os.path.join(root_dir, 'annotation')
    if not os.path.exists(test_pose_save_dir):
        os.makedirs(test_pose_save_dir)

    label_name = 'test.csv'
    label_filename = os.path.join(label_dir, label_name)
    df = pd.read_csv(label_filename)

    for i in range(0, len(df)):
        filename = df.loc[i, 'name']

        video_path = os.path.join(test_video_dir, filename)
        test_pose_save_path = os.path.join(test_pose_save_dir, filename.replace('mp4', 'npy'))
        print('\nvideo input path:', video_path)
        print('test pose save path:', test_pose_save_path)

        video_cap = cv2.VideoCapture(video_path)
        # Get some video parameters.
        video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize tracker.
        pose_tracker = mp_pose.Pose()

        np_pose = []
        while True:
            # Get next frame of the video.
            success, frame = video_cap.read()
            if not success:
                break
            # Run pose tracker.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose_tracker.process(image=frame)
            pose_landmarks = result.pose_landmarks

            if pose_landmarks is not None:
                pose_landmarks = np.array(
                    [[lmk.x * video_width, lmk.y * video_height, lmk.z * video_width]
                     for lmk in pose_landmarks.landmark],
                    dtype=np.float32)

                # Calculate the 5 left joint angles
                shoulder_left = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                elbow_left = pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                wrist_left = pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                hip_left = pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                knee_left = pose_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                ankle_left = pose_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

                elbow_angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
                shoulder_angle_left = calculate_angle(hip_left, shoulder_left, elbow_left)
                hip_angle_left = calculate_angle(shoulder_left, hip_left, knee_left)
                knee_angle_left = calculate_angle(hip_left, knee_left, ankle_left)
                ankle_angle_left = calculate_angle(knee_left, ankle_left,
                                                   pose_landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value])

                # Calculate the 5 right joint angles
                shoulder_right = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                elbow_right = pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                wrist_right = pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                hip_right = pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                knee_right = pose_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                ankle_right = pose_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

                elbow_angle_right = calculate_angle(shoulder_right, elbow_right, wrist_right)
                shoulder_angle_right = calculate_angle(hip_right, shoulder_right, elbow_right)
                hip_angle_right = calculate_angle(shoulder_right, hip_right, knee_right)
                knee_angle_right = calculate_angle(hip_right, knee_right, ankle_right)
                ankle_angle_right = calculate_angle(knee_right, ankle_right, pose_landmarks[
                    mp_pose.PoseLandmark.RIGHT_HEEL.value])

                # calculate average angles of left and right
                average_elbow_angle = calculate_average_angle(elbow_angle_left, elbow_angle_right)
                average_shoulder_angle = calculate_average_angle(shoulder_angle_left,
                                                                 shoulder_angle_right)
                average_hip_angle = calculate_average_angle(hip_angle_left, hip_angle_right)
                average_knee_angle = calculate_average_angle(knee_angle_left, knee_angle_right)
                average_ankle_angle = calculate_average_angle(ankle_angle_left, ankle_angle_right)

                # Normalize the angles
                all_angle_landmarks = [average_elbow_angle, average_shoulder_angle, average_hip_angle,
                                            average_knee_angle, average_ankle_angle]

                all_angle_landmarks = np.array(all_angle_landmarks, np.float32)
                train_angle_landmarks = normalize_angles_horizontal(all_angle_landmarks)
                print(f'train_angles.shape is {train_angle_landmarks.shape}')

                if args.input == "only_angles":
                    # padding the angles with 0
                    # desired_length = 99
                    # print(train_angle_landmarks.shape[0])
                    # landmarks = np.pad(train_angle_landmarks, (0, desired_length - train_angle_landmarks.shape[0]),
                    #                          mode='constant')

                    # rep the angles
                    repetitions = 20
                    landmarks = np.tile(train_angle_landmarks, (repetitions, ))
                    print(landmarks.shape)
                else:
                    lanrmarks = np.expand_dims(pose_landmarks, axis=0)
                    landmarks = normalize_landmarks(lanrmarks)

                    # Append the angle values to the pose_landmarks
                    landmarks = np.append(landmarks, train_angle_landmarks)

                    landmarks = np.array(landmarks).astype(np.float32).reshape(-1)
                    print(landmarks.shape)
            else:
                landmarks = np.zeros(104)  # 99 99 for only five angles with padding, 104
                print(landmarks.shape)
            np_pose.append(landmarks)
            print(len(np_pose))
        np_pose = np.array(np_pose).astype(np.float32)
        np.save(test_pose_save_path, np_pose)

    current_time = time.time()
    print('time: ' + str(current_time - pre_time) + 's')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate our PoseRAC')
    parser.add_argument('--config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('--input', type=str, metavar='DIR',
                        help=' "only angles" or coordinates and angles')
    parser.add_argument('--output', type=str, metavar='DIR',
                        help='output dir angle npy files, such as test_poses_only_5_ave_rep20')
    args = parser.parse_args()
    main(args)
