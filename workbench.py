import h5py,os
from utils.config import HParameters
from utils import Proportion
import torch
import models
from torchsummary import summary
from models.rand import RandomTrainer
from models.logistic import LogisticRegressionTrainer
from models.vasnet import VASNetTrainer
from models.transformer import TransformerTrainer, Transformer
from models.dsn import DSNTrainer
from models.sumgan import SumGANTrainer
from models.sumgan_att import SumGANAttTrainer
# filename = 'sample_feature/sample_GoogleNet.h5'

# filename = 'logs/1707737585_TransformerTrainer/summe_splits.json_preds.h5'
# def infer(filename = 'sample_feature/sample_GoogleNet.h5'):
#     data = None
#     with h5py.File(filename, 'r') as h5file:
#         for k in h5file.keys():
#             data = h5file[k][:]
#             data = torch.from_numpy(data)
#             with torch.no_grad():
#                 pred = model.model(data[None, ...])
#                 print(f'{k} -> x_dims {data.size()} -> y_dims {pred.size()}')

def read_h5_file(filename):
    with h5py.File(filename,'r') as file:
        videos = (list(file.keys()))
        frames = []
        for key in videos:
            # print(list(file[key].keys()))
            # print(file[key]['video_name'][()])
            # print(file[key]['n_frames'][()])
            # print(sum(list(file[key]['n_frame_per_seg'][:])))
            frames.append(file[key]['n_frames'][()])
    return frames, videos
            

def check_frames(file_path):
    frames = os.listdir(file_path)
    frames = [i for i in frames if i.endswith("jpg")]
    return len(frames)
# hps = HParameters()
# print(hps)
# hps.extra_params = {'encoder_layers': 6}
# def load_model(model_weights_path):
#     # Load model weights
#     model_state_dict = torch.load(model_weights_path)
#     # Create an instance of the model class
#     model = TransformerTrainer(hps, splits_file='splits/summe_splits.json')
#     # model.load_state_dict(model_state_dict)
#     # Set model to evaluation mode
#     # model.eval()
#     return model

import cv2
import os
import numpy as np
import glob

def mse(imageA, imageB):
    # Compute the Mean Squared Error between two images
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def is_frame_match(frame1, frame2, threshold=1000):
    # Compare frames using MSE
    error = mse(frame1, frame2)
    return error < threshold

def match_frame_in_video(video_path, frame_path, threshold=1000):
    # Load the frame to match
    frame_to_match = cv2.imread(frame_path)
    frame_to_match_gray = cv2.cvtColor(frame_to_match, cv2.COLOR_BGR2GRAY)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {video_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Check if the current frame matches the frame to match
        if is_frame_match(frame_to_match_gray, frame_gray, threshold):
            cap.release()
            return True

        frame_count += 1

    cap.release()
    return False

def find_matching_video(video_directory, frame_path, threshold=1000):
    video_files = glob.glob(os.path.join(video_directory, "*.mp4"))
    
    for video_path in video_files:
        print(f"Checking video: {video_path}")
        if match_frame_in_video(video_path, frame_path, threshold):
            return os.path.basename(video_path)
    
    return None



# model_weights_path = '/mnt/g/Github/video_summarizer/logs/1707737585_TransformerTrainer/summe_splits.json.pth'
# model = load_model(model_weights_path)
# # model.load_weights(model_weights_path)
# print(f'model loaded successfully from {model_weights_path}')
# infer()
if __name__ == "__main__":
    filename = 'datasets/summarizer_dataset_tvsum_google_pool5.h5'
    h5frames, h5videos = read_h5_file(filename)
    dict = {}
    for i in range(len(h5videos)):
        dict[h5videos[i]] = h5frames[i]
    # print(h5frames)
    # print(len(h5frames), len(set(h5frames)))
    # file_path = '/mnt/g/Github/video_summarizer/datasets/video/_frames/video_10'
    # check_frames(file_path)
    video_directory = 'datasets/video/_frames/'
    new_video_directory = 'datasets/video/frames/'
    folders = os.listdir(video_directory)
    data_frames = []
    for folder in folders:
        file_path = os.path.join(video_directory, folder)
        data_frames.append(check_frames(file_path))
    # print(data_frames)
    # print(len(data_frames), len(set(data_frames)))  
    matches = set(h5frames).intersection(set(data_frames))
    # print(len(matches), matches)
    nonmatches1 = set(h5frames).difference(set(data_frames))
    nonmatches2 = set(data_frames).difference(set(h5frames))
    # print(len(nonmatches1), nonmatches1)
    # print(len(nonmatches2), nonmatches2)
    inverted_dict = {v: k for k, v in dict.items()}
    for folder in folders:
        # check the matches set and is the data frames in the matches set then rename the folder with h5 videos
        file_path = os.path.join(video_directory, folder)
        frames = check_frames(file_path)

        if frames in matches:
            print(file_path, os.path.join(new_video_directory, f'{inverted_dict[frames]}'))
            # os.rename(file_path, os.path.join(new_video_directory, f'{inverted_dict[frames]}'))
        else:
            # check the nearest match, nearest int in the nonmatches1
            nearest = min(nonmatches1, key=lambda x:abs(x-frames))
            print("---nonmatches---",nearest, frames)
            print(file_path, os.path.join(new_video_directory, f'{inverted_dict[nearest]}'))
            # os.rename(file_path, os.path.join(new_video_directory, f'{inverted_dict[nearest]}'))
            print("------------")
    # video_directory = 'path_to_your_video_directory'
    # frame_path = 'path_to_your_frame.jpg'

    # matching_video = find_matching_video(video_directory, frame_path)

    # if matching_video:
    #     print(f"The frame matches with the video: {matching_video}")
    # else:
    #     print("No matching video found.")
'''
datasets/video/_frames/video_1 datasets/video/frames/video_50
datasets/video/_frames/video_10 datasets/video/frames/video_46
datasets/video/_frames/video_11 datasets/video/frames/video_38
datasets/video/_frames/video_12 datasets/video/frames/video_42
datasets/video/_frames/video_13 datasets/video/frames/video_24
datasets/video/_frames/video_14 datasets/video/frames/video_6
datasets/video/_frames/video_15 datasets/video/frames/video_17
datasets/video/_frames/video_16 datasets/video/frames/video_3
datasets/video/_frames/video_17 datasets/video/frames/video_32
datasets/video/_frames/video_18 datasets/video/frames/video_44
datasets/video/_frames/video_19 datasets/video/frames/video_20
datasets/video/_frames/video_2 datasets/video/frames/video_13
datasets/video/_frames/video_20 datasets/video/frames/video_47
datasets/video/_frames/video_21 datasets/video/frames/video_25
datasets/video/_frames/video_22 datasets/video/frames/video_27
---nonmatches--- 4165 4166
datasets/video/_frames/video_23 datasets/video/frames/video_39
------------
datasets/video/_frames/video_24 datasets/video/frames/video_31
---nonmatches--- 9534 9535
datasets/video/_frames/video_25 datasets/video/frames/video_16
------------
datasets/video/_frames/video_26 datasets/video/frames/video_36
datasets/video/_frames/video_27 datasets/video/frames/video_23
datasets/video/_frames/video_28 datasets/video/frames/video_5
datasets/video/_frames/video_29 datasets/video/frames/video_18
datasets/video/_frames/video_3 datasets/video/frames/video_19
datasets/video/_frames/video_30 datasets/video/frames/video_35
datasets/video/_frames/video_31 datasets/video/frames/video_10
datasets/video/_frames/video_32 datasets/video/frames/video_22
datasets/video/_frames/video_33 datasets/video/frames/video_34
datasets/video/_frames/video_34 datasets/video/frames/video_21
datasets/video/_frames/video_35 datasets/video/frames/video_43
datasets/video/_frames/video_36 datasets/video/frames/video_29
datasets/video/_frames/video_37 datasets/video/frames/video_4
datasets/video/_frames/video_38 datasets/video/frames/video_11
datasets/video/_frames/video_39 datasets/video/frames/video_45
datasets/video/_frames/video_4 datasets/video/frames/video_14
datasets/video/_frames/video_40 datasets/video/frames/video_49
datasets/video/_frames/video_41 datasets/video/frames/video_48
datasets/video/_frames/video_42 datasets/video/frames/video_40
datasets/video/_frames/video_43 datasets/video/frames/video_41
datasets/video/_frames/video_44 datasets/video/frames/video_7
datasets/video/_frames/video_45 datasets/video/frames/video_37
datasets/video/_frames/video_46 datasets/video/frames/video_8
datasets/video/_frames/video_47 datasets/video/frames/video_33
datasets/video/_frames/video_48 datasets/video/frames/video_9
datasets/video/_frames/video_49 datasets/video/frames/video_15
datasets/video/_frames/video_5 datasets/video/frames/video_30
datasets/video/_frames/video_50 datasets/video/frames/video_28
datasets/video/_frames/video_6 datasets/video/frames/video_26
datasets/video/_frames/video_7 datasets/video/frames/video_2
datasets/video/_frames/video_8 datasets/video/frames/video_1
datasets/video/_frames/video_9 datasets/video/frames/video_12
'''