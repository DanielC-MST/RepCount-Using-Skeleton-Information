# _*_ coding: utf-8 _*_
# @Author: Haodong_Chen
# @Time: 7/12/23 1:47 PM
import pandas as pd
import numpy as np
import os
import csv
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
from model import PoseRAC
import argparse
import time
import yaml
import matplotlib.pyplot as plt
torch.multiprocessing.set_sharing_strategy('file_system')


# Normalization of angles to improve training robustness.
def normalize_angles(angles_landmarks):
    normalized_angle_landmarks = np.copy(angles_landmarks)
    print(len(angles_landmarks))

    for i in range(angles_landmarks.shape[1]):
        column = angles_landmarks[:, i]
        max_value = np.max(column)
        min_value = np.min(column)

        normalized_angle_landmarks[:, i] = (column - min_value) / (max_value - min_value)

    return normalized_angle_landmarks


# Normalization to improve training robustness.
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

    all_landmarks = all_landmarks.reshape(len(all_landmarks), 99)
    return all_landmarks


# For each pose, we use 33 key points to represent it, and each key point has 3 dimensions xyz.
# Obtain the pose information (33*3 + 5 = 104) of each key frame, and set up the label
# 1 for salient pose I and 0 for salient pose II.
def obtain_landmark_label(csv_path, all_landmarks, all_labels, label2index, num_classes):
    file_separator = ','
    n_landmarks = 33
    n_dimensions = 3
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=file_separator)
        for row in csv_reader:
            # assert len(row) == n_landmarks * n_dimensions + 2, 'Wrong number of values: {}'.format(len(row))
            # landmarks = np.array(row[2:], np.float32).reshape([n_landmarks, n_dimensions])
            assert len(row) == n_landmarks * n_dimensions + 7, 'Wrong number of values: {}'.format(len(row))
            landmarks = np.array(row[2:101], np.float32).reshape([n_landmarks, n_dimensions])

            all_landmarks.append(landmarks)
            label = label2index[row[1]]

            start_str = row[0].split('/')[-3]
            label_np = np.zeros(num_classes)
            if start_str == 'salient1':
                # if salient2 pose happens, then label_np = 0
                label_np[label] = 1
            all_labels.append(label_np)
    return all_landmarks, all_labels


def csv2data(train_csv, action2index, num_classes):
    train_landmarks = []
    train_labels = []
    train_landmarks, train_labels = obtain_landmark_label(train_csv, train_landmarks, train_labels, action2index, num_classes)

    train_landmarks = np.array(train_landmarks)
    train_labels = np.array(train_labels)
    train_landmarks = normalize_landmarks(train_landmarks)

    all_angle_landmarks = []
    with open(train_csv) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            all_angle_landmarks.append(row[-5:]) # 10 means 10 joint angles
    all_angle_landmarks = np.array(all_angle_landmarks, np.float32)
    train_angle_landmarks = normalize_angles(all_angle_landmarks)
    train_landmarks = np.concatenate((train_landmarks, train_angle_landmarks), axis=1)
    print(f'train_landmarks.shape is {train_landmarks.shape}')
    return train_landmarks, train_labels


def obtain_landmark_label_only_angles(csv_path, all_labels, label2index, num_classes):
    file_separator = ','
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=file_separator)
        for row in csv_reader:
            label = label2index[row[1]]
            start_str = row[0].split('/')[-3]
            label_np = np.zeros(num_classes)
            if start_str == 'salient1':
                label_np[label] = 1 # if salient2 pose, then label_np = 0
            all_labels.append(label_np)
    return all_labels


def csv2data_only_angles(args, train_csv, action2index, num_classes):
    train_labels = []
    train_labels = obtain_landmark_label_only_angles(train_csv, train_labels, action2index, num_classes)
    train_labels = np.array(train_labels)
    # train_angles = []
    all_angle_landmarks = []
    with open(train_csv) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            all_angle_landmarks.append(row[-5:])
    all_angle_landmarks = np.array(all_angle_landmarks, np.float32)
    train_angle_landmarks = normalize_angles(all_angle_landmarks)
    print(f'train_angles.shape is {train_angle_landmarks.shape}')
    # desired_length = 100
    repetitions = 20
    train_landmarks = np.tile(train_angle_landmarks, (1, repetitions))
    print(train_landmarks.shape)

    print(f'train_landmarks.shape is {train_landmarks.shape}')
    return train_landmarks, train_labels


def main(args):
    old_time = time.time()
    if os.path.isfile(args.config):
        with open(args.config, "r") as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)
    else:
        raise ValueError("Config file does not exist.")

    csv_label_path = config['dataset']['csv_label_path'] # 8 actions, all_action.csv
    root_dir = config['dataset']['dataset_root_dir']

    # key frame file + action type + (x,y,z) * 33 = 101
    train_csv = os.path.join(root_dir, 'annotation_pose', 'train_angle_5_ave.csv')

    label_pd = pd.read_csv(csv_label_path) # 8 actions
    index_label_dict = {}
    length_label = len(label_pd.index) # 8
    for label_i in range(length_label):
        one_data = label_pd.iloc[label_i]
        action = one_data['action']
        label = one_data['label']
        index_label_dict[label] = action
    num_classes = len(index_label_dict) # 8
    action2index = {v: k for k, v in index_label_dict.items()}

    if args.input == 'only_angles':
        train_landmarks, train_labels = csv2data_only_angles(args, train_csv, action2index, num_classes)
        valid_landmarks, valid_labels = csv2data_only_angles(args, train_csv, action2index, num_classes)
    else:
        train_landmarks, train_labels = csv2data(train_csv, action2index, num_classes)
        valid_landmarks, valid_labels = csv2data(train_csv, action2index, num_classes)
    print(f'the train_landmarks shape is {train_landmarks.shape}')
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=20,
        verbose=True,
        mode='min',
    )
    ckpt_callback = ModelCheckpoint(mode="min",
                                    monitor="val_loss",
                                    dirpath='./saved_weights/' + args.saved_weights_dir + '/',
                                    filename='{epoch}-{val_loss:.2f}',
                                    every_n_epochs=1,
                                    save_top_k=-1)

    model = PoseRAC(train_landmarks, train_labels, valid_landmarks, valid_labels, dim=config['PoseRAC']['dim'],
                    heads=config['PoseRAC']['heads'], enc_layer=config['PoseRAC']['enc_layer'],
                    learning_rate=config['PoseRAC']['learning_rate'], seed=config['PoseRAC']['seed'],
                    num_classes=num_classes, alpha=config['PoseRAC']['alpha'])
    print(config['trainer']['auto_lr_find'])
    trainer = pl.Trainer(callbacks=[early_stop_callback, ckpt_callback], max_epochs=config['trainer']['max_epochs'],
                         auto_lr_find=config['trainer']['auto_lr_find'], accelerator=config['trainer']['accelerator'],
                         devices=config['trainer']['devices'], strategy='ddp')

    trainer.tune(model)
    print('Learning rate:', model.learning_rate)

    trainer.fit(model)

    print(f'best loss: {ckpt_callback.best_model_score.item():.5g}')

    weights = model.state_dict()
    torch.save(weights, config['save_ckpt_path'])

    current_time = time.time()
    print('time: ' + str(current_time - old_time) + 's')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate our PoseRAC')
    parser.add_argument('--config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('--input', type=str, metavar='DIR',
                        help='inputs are "only angles" or both 33 points coordinates and 5 angles')
    parser.add_argument('--saved_weights_dir', type=str, metavar='DIR',
                        help='the dir saving all weights')
    args = parser.parse_args()
    main(args)
