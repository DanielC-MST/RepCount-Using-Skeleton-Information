import os
import cv2
import csv
from mediapipe.python.solutions import pose as mp_pose
import numpy as np


# Calculate the joint angles
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


def calculate_average_angle(left_angles, right_angles):
    average_angles = (left_angles + right_angles) / 2
    return average_angles


# For the selected key frames, we use the pose estimation network to extract the poses.
# For each pose, we use 33 key points to represent it, and each key point has 3 dimensions.
def _generate_for_train_angle(root_dir):
    data_folder = os.path.join(root_dir, 'extracted')
    out_csv_dir = os.path.join(root_dir, 'annotation_pose')

    if not os.path.exists(out_csv_dir):
        os.makedirs(out_csv_dir)

    for train_type in os.listdir(data_folder):
        if '.DS_Store' in train_type:
            continue
        out_csv_path = os.path.join(out_csv_dir, train_type) + '_angle_5average.csv'
        with open(out_csv_path, 'w') as csv_out_file:
            csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            sub_train_folder = os.path.join(data_folder, train_type)
            for action_type in os.listdir(sub_train_folder):
                sub_sub_folder = os.path.join(sub_train_folder, action_type)
                print(action_type)
                if '.DS_Store' in action_type:
                    continue
                for salient1_2 in os.listdir(sub_sub_folder):
                    sub_sub_sub_folder = os.path.join(sub_sub_folder, salient1_2)
                    if '.DS_Store' in salient1_2:
                        continue
                    for video_name in os.listdir(sub_sub_sub_folder):
                        video_dir = os.path.join(sub_sub_sub_folder, video_name)
                        if '.DS_Store' in video_dir:
                            continue
                        for single_path in os.listdir(video_dir):
                            if '.DS_Store' in single_path:
                                continue
                            if '.jpg' not in single_path:
                                continue
                            image_path = os.path.join(video_dir, single_path)
                            base_path = os.path.join(train_type, action_type, salient1_2, video_name, single_path)
                            input_frame = cv2.imread(image_path)
                            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

                            # Initialize fresh pose tracker and run it.
                            with mp_pose.Pose() as pose_tracker:
                                result = pose_tracker.process(image=input_frame)
                                pose_landmarks = result.pose_landmarks
                            output_frame = input_frame.copy()
                            # Save landmarks if pose was detected.
                            if pose_landmarks is not None:
                                # Get landmarks.
                                frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                                pose_landmarks = np.array(
                                    [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                     for lmk in pose_landmarks.landmark],
                                    dtype=np.float32)
                                assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(
                                    pose_landmarks.shape)
                                # Calculate the 5 left joint angles
                                shoulder_left = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                                elbow_left = pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                                wrist_left = pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                                hip_left = pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                                knee_left = pose_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                                ankle_left = pose_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

                                # Calculate the 5 right joint angles
                                shoulder_right = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                                elbow_right = pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                                wrist_right = pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                                hip_right = pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                                knee_right = pose_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                                ankle_right = pose_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

                                elbow_angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
                                shoulder_angle_left = calculate_angle(hip_left, shoulder_left, elbow_left)
                                hip_angle_left = calculate_angle(shoulder_left, hip_left, knee_left)
                                knee_angle_left = calculate_angle(hip_left, knee_left, ankle_left)
                                ankle_angle_left = calculate_angle(knee_left, ankle_left,
                                                                   pose_landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value])

                                elbow_angle_right = calculate_angle(shoulder_right, elbow_right, wrist_right)
                                shoulder_angle_right = calculate_angle(hip_right, shoulder_right, elbow_right)
                                hip_angle_right = calculate_angle(shoulder_right, hip_right, knee_right)
                                knee_angle_right = calculate_angle(hip_right, knee_right, ankle_right)
                                ankle_angle_right = calculate_angle(knee_right, ankle_right, pose_landmarks[
                                    mp_pose.PoseLandmark.RIGHT_HEEL.value])

                                average_elbow_angle = calculate_average_angle(elbow_angle_left, elbow_angle_right)
                                average_shoulder_angle = calculate_average_angle(shoulder_angle_left,
                                                                                 shoulder_angle_right)
                                average_hip_angle = calculate_average_angle(hip_angle_left, hip_angle_right)
                                average_knee_angle = calculate_average_angle(knee_angle_left, knee_angle_right)
                                average_ankle_angle = calculate_average_angle(ankle_angle_left, ankle_angle_right)

                                # Append the angle values to the pose_landmarks
                                pose_landmarks = np.append(pose_landmarks,
                                                           [average_elbow_angle, average_shoulder_angle, average_hip_angle,
                                                            average_knee_angle, average_ankle_angle])

                                # elbow_angle = calculate_angle(shoulder, elbow, wrist)
                                # shoulder_angle = calculate_angle(hip, shoulder, elbow)
                                # hip_angle = calculate_angle(shoulder, hip, knee)
                                # knee_angle = calculate_angle(hip, knee, ankle)
                                # ankle_angle = calculate_angle(knee, ankle, pose_landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value])

                                # Append the angle values to the pose_landmarks
                                # pose_landmarks = np.append(pose_landmarks, [elbow_angle, shoulder_angle, hip_angle, knee_angle, ankle_angle])

                                # Convert pose_landmarks to a flattened string format
                                pose_landmarks_str = pose_landmarks.flatten().astype(str).tolist()

                                # Write the base_path, action_type, and pose_landmarks to the CSV file
                                csv_out_writer.writerow([base_path, action_type] + pose_landmarks_str)
