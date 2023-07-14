# Peloton-Intern-RepCount
RepCount using skeleton and joint information

utils folder done

#### Download Videos and Pose-level Annotations
[this Google Drive link](https://drive.google.com/file/d/1k9LLzOsJVh6ACXSX8iKbGNxTY9-L6X_x/view?usp=sharing)

## Code overview
* After preparing the dataset above, the folder structure should look like:
```
This folder
│   README.md
│   best_weights.pth
│   pre_train_angles.py
|   train.py
│   pre_test_angles.py
|   eval.py
│   ...

└───RepCount_pose/
│    └───annotation/
│    │	 └───pose_train.csv
│    │	 └───test.csv  
│    │   └───valid.csv 
│    │   └───video_train.csv
│    └───original_data/
│    └───annotation_pose/
│    │	 └───train.csv
│    │	 └───train_angle.csv  
│    │   └───train_angle_5_ave.csv 
│    └───test_poses_5_ave/
│    └───video/
│    │	 └───test/
│    │	 └───train/
│    │   └───valid/
│    └───video_visual_output/
│    │   └───test_density_img_5_ave/

```

## Code
### Install
Please refer to INSTALL.md for installation, or you can use:
```
pip install -r requirement.txt
```

### Evaluation
- [**Optional**] Extrace the poses and joint angles (33*3 + 5) for each frame of all test videos. 
- As all poses of the test videos have been already extracted by us (see the *./RepCount_pose/test_poses_5_ave/*).
- If you wnat to extract by yourself, you can try this to generate all the data in *./RepCount_pose/test_poses_5_ave/*.

```
python pre_test_angles.py --config ./RepCount_pose_config.yaml --input coor_and_angle --output test_poses_5_ave
```


- Evaluate our PoseRAC with pretrained checkpoint:
```
python eval.py --config ./RepCount_pose_config.yaml --pth_dir ./best_weights.pth --test_pose_save_dir test_poses_5_ave --test_video_dir video/test

```
- Then, you can get the results:
```
MAE:0.2108900693807817, OBO:0.5921052631578947

```

### Training
- Preprocessing before training. According to the pose-level annotation, we extract the salient poses of the salient frames and obtain their corresponding classes.

```
python pre_train_angles.py --config ./RepCount_pose_config.yaml
```

- Train the model:
```
python train.py --config ./RepCount_pose_config.yaml --input coor_and_angle --saved_weights_dir saved_all_weights_5_ave
```

### Inference and Visualization
```
python inference_and_visualization_all.py --config ./RepCount_pose_config.yaml --pth ./new_weights.pth
```

You can also train from scratch to get a set of model weights for evaluation and inference.

## Contact
Haodong Chen (h.chen@mst.edu)

If you have any questions or suggestions, don't hesitate to contact me!

