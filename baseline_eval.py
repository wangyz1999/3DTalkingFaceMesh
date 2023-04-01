import argparse
import numpy as np
import os

import torch
import torch.nn.functional as F

from utils.drawing_utils import lips_idx, inner_lips_idx

lips_lm = lips_idx + inner_lips_idx

parser = argparse.ArgumentParser(description='Speech-driven Talking 3D Face Mesh using Transformer')
parser.add_argument("--seed", type=int, default=44, help='seed of experiment')
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--audio_path", type=str, default=r"data\audios", help='path of audio data')
parser.add_argument("--demo_audio_path", type=str, default=r"data\demo_audio", help='path of audio data')
parser.add_argument("--landmark_path", type=str, default=r"data\landmarks", help='path of landmark data')
parser.add_argument("--validity_path", type=str, default=r"data\validity", help='path of validity data')
parser.add_argument("--ref_face_path", type=str, default=r"data\close_mouth_face.npy", help='path of validity data')
parser.add_argument("--data_split_path", type=str, default=r"data\train_test_split.txt", help='path of data split')
args = parser.parse_args()


close_mouth_face = np.load(r"data\close_mouth_face.npy").astype("float32")
close_mouth_face = torch.from_numpy(close_mouth_face)

train_audio, valid_audio, test_audio = [], [], []
train_displace, valid_displace, test_displace = [], [], []
file_id_type = [tuple(fid.split('\t')) for fid in open(args.data_split_path).read().split('\n')]

all_landmarks = []
for f, data_type in file_id_type:
    if data_type != 'test':
        continue

    landmark_array = np.load(os.path.join(args.landmark_path, f)).astype("float32")
    validity_array = np.load(os.path.join(args.validity_path, f))
    landmark_array[~validity_array] = close_mouth_face
    all_landmarks.append(torch.from_numpy(landmark_array))
all_landmarks = torch.vstack(all_landmarks)
mean_face = all_landmarks.mean(dim=0)

close_mouth_face = close_mouth_face[None, :].repeat(all_landmarks.shape[0], 1, 1)
mean_face = mean_face[None, :].repeat(all_landmarks.shape[0], 1, 1)



for ref_face in [close_mouth_face, mean_face]:
    landmark_vel = torch.diff(all_landmarks, dim=1)
    ref_vel = torch.diff(ref_face, dim=1)

    face_mse = F.mse_loss(all_landmarks.flatten(start_dim=1), ref_face.flatten(start_dim=1), reduction='none').mean(dim=1).detach().cpu().numpy()
    lips_mse = F.mse_loss(all_landmarks[..., lips_lm].flatten(start_dim=1),
                          ref_face[..., lips_lm].flatten(start_dim=1), reduction='none').mean(
        dim=1).detach().cpu().numpy()
    face_vel_mse = F.mse_loss(landmark_vel.flatten(start_dim=1), ref_vel.flatten(start_dim=1),
                              reduction='none').mean(dim=1).detach().cpu().numpy()
    lips_vel_mse = F.mse_loss(landmark_vel[..., lips_lm].flatten(start_dim=1),
                              ref_vel[..., lips_lm].flatten(start_dim=1), reduction='none').mean(
        dim=1).detach().cpu().numpy()

    face_mse_mean = face_mse.mean()
    face_mse_std = face_mse.std()
    lips_mse_mean = lips_mse.mean()
    lips_mse_std = lips_mse.std()

    face_vel_mse_mean = face_vel_mse.mean()
    face_vel_mse_std = face_vel_mse.std()
    lips_vel_mse_mean = lips_vel_mse.mean()
    lips_vel_mse_std = lips_vel_mse.std()

    result = {
        'face_mean': face_mse_mean,
        'face_std': face_mse_std,
        'lips_mean': lips_mse_mean,
        'lips_std': lips_mse_std,
        'face_vel_mean': face_vel_mse_mean,
        'face_vel_std': face_vel_mse_std,
        'lips_vel_mean': lips_vel_mse_mean,
        'lips_vel_std': lips_vel_mse_std,
    }
    print([
        result['face_mean'],
        result['face_std'],
        result['lips_mean'],
        result['lips_std'],
        result['face_vel_mean'],
        result['face_vel_std'],
        result['lips_vel_mean'],
        result['lips_vel_std'],
    ])
