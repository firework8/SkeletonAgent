import os
import json
import pickle
import argparse

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

import mmcv
import sys


CS_train_V1 = [0, 2, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                21, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 59, 
                61, 62, 63, 64, 65, 67, 68, 69, 70, 71, 73, 76, 77, 78, 79, 80, 
                81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 98, 100, 102, 103, 105, 
                106, 110, 111, 112, 114, 115, 116, 117, 118]

CS_train_V2 = [0, 3, 4, 5, 6, 8, 10, 11, 12, 14, 16, 18, 19, 20, 21, 22, 24, 
                26, 29, 30, 31, 32, 35, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47, 
                49, 52, 54, 56, 57, 59, 60, 61, 62, 63, 64, 66, 67, 69, 70, 71, 
                72, 73, 74, 75, 77, 78, 79, 80, 81, 83, 84, 86, 87, 88, 89, 91, 
                92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 106, 107, 
                108, 109, 110, 111, 112, 113, 114, 115, 117, 118]


def extract_pose(ske_txt_path: str) -> np.ndarray:
    with open(ske_txt_path, 'r') as f: 
        num_frame = int(f.readline()) # the frame num
        joint_data = [] # T M V C
        for t in range(num_frame): # for each frame
            num_body = int(f.readline()) # the body num
            one_frame_data = np.zeros((num_body, 17, 2)) # M 17 2 
            for m in range(num_body): # for each body
                f.readline() # skip this line, e.g. 000 0 0 0 0 0 0 0 0 0
                num_joints = int(f.readline()) # the num joins, equal to 17
                assert num_joints == 17
                for v in range(num_joints): # for each joint
                    xy = np.array(f.readline().split()[:2], dtype = np.float64)
                    one_frame_data[m, v] = xy
            joint_data.append(one_frame_data)
        joint_data = np.array(joint_data)  
    return joint_data # T M 17 2 


def get_max_frame(root_Skeleton_path:str, samples_txt: list) -> int:
    max_frame = 0
    for idx, sample in enumerate(samples_txt):
        ske_path = root_Skeleton_path + '/' + sample
        with open(ske_path, 'r') as f:
            cur_frame = int(f.readline()) # the frame num
            if cur_frame > max_frame: max_frame = cur_frame
    return max_frame 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='UAV-Human Data Converter.')
    parser.add_argument(
        '--data_path', default='/data/uav-human-skeleton/Skeleton')
    parser.add_argument(
        '--out_folder', default='/data/uav')
    arg = parser.parse_args()

    if not os.path.exists(arg.out_folder):
        os.makedirs(arg.out_folder)
    
    root_Skeleton_path = arg.data_path
    samples_txt = sorted(os.listdir(root_Skeleton_path))
    
    CSv1_train_name = []
    CSv1_val_name = []
    CSv2_train_name = []
    CSv2_val_name = []
    all_annotations = []
    
    for idx, sample in enumerate(tqdm(samples_txt)):
        
        frame_dir = sample[:-4]
        subj_id = int(sample[1:4]) # get subject id
        label_id = int(sample.split('A')[1][:3])
        label = label_id
        ske_path = root_Skeleton_path + '/' + sample
        joint_data = extract_pose(ske_path)
        
        # T M V C -> M T V C
        keypoint = joint_data.transpose(1, 0, 2, 3)
        score_shape = (keypoint.shape[0], keypoint.shape[1], keypoint.shape[2])
        keypoint_score = np.ones(score_shape, dtype=np.float32)
        total_frames = keypoint.shape[1]
        
        annotation = {'frame_dir': frame_dir, 'label': label, 'keypoint': keypoint, 'keypoint_score': keypoint_score, 'total_frames': total_frames}
        all_annotations.append(annotation)
        
        if subj_id in CS_train_V1:
            CSv1_train_name.append(frame_dir)
        else:
            CSv1_val_name.append(frame_dir)
        
        if subj_id in CS_train_V2:
            CSv2_train_name.append(frame_dir)
        else:
            CSv2_val_name.append(frame_dir)
    
    split = {'csv1_train': CSv1_train_name, 'csv1_val': CSv1_val_name, 'csv2_train': CSv2_train_name, 'csv2_val': CSv2_val_name}
    data = {'split': split, 'annotations': all_annotations}
    
    mmcv.dump(data, '/data/uav/uav_human.pkl')
    print("success")
    