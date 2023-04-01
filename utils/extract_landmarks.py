import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import os
# import subprocess as sp
# import shlex
# import json
import argparse
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from multiprocessing import Pool, cpu_count
matplotlib.use('TkAgg')

from videosource import FileSource
from face_geometry import (  # isort:skip
    PCF,
    get_metric_landmarks,
)

num_cpus = cpu_count()

def prepare_pcf(frame_width, frame_height):
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )
    pcf = PCF(
        near=1,
        far=10000,
        frame_height=frame_height,
        frame_width=frame_width,
        fy=camera_matrix[1, 1],
    )
    return pcf


def get_landmark_from_video(video_file, face_mesh):
    """
    Args:
        video_file: the video file that is going to get landmarks extracted
        face_mesh: mediapipe face mesh extractor
    Returns:
        landmarks: the screen 3d landmarks
        metric_landmarks: the metric 3d landmarks
        has_missing_face: There is missing face in at least one frame of the video
        has_multiple_face: Multiple faces got detected in at least one frame of the video
    """
    source = FileSource(video_file)

    width, height = source.get_image_size()
    pcf = prepare_pcf(width, height)

    all_metric_landmarks = []
    valid_frames = []
    for idx, (frame, frame_rgb) in enumerate(source):
        results = face_mesh.process(frame_rgb)
        multi_face_landmarks = results.multi_face_landmarks

        if multi_face_landmarks and len(multi_face_landmarks) == 1:
            face_landmarks = multi_face_landmarks[0]
            landmarks = np.array(
                [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
            )
            landmarks = landmarks.T
            landmarks = landmarks[:, :468]

            metric_landmarks, pose_transform_mat_ = get_metric_landmarks(
                landmarks.copy(), pcf
            )
            all_metric_landmarks.append(metric_landmarks)
            valid_frames.append(True)
        else:
            all_metric_landmarks.append(np.zeros((3, 468)))
            valid_frames.append(False)
    valid_frames = np.array(valid_frames)
    return all_metric_landmarks, valid_frames


def process_video_files(files, video_folder, landmark_folder, validity_folder):
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    for video in tqdm(files):
        video_file = os.path.join(video_folder, video)
        metric_landmarks, valid_frames = get_landmark_from_video(video_file, face_mesh)
        video_file_name = video.split('.')[0]
        np.save(os.path.join(landmark_folder, video_file_name), metric_landmarks)
        np.save(os.path.join(validity_folder, video_file_name), valid_frames)


def main():
    video_folder = "videos_25"
    landmark_folder = "landmarks"
    validity_folder = "validity"
    all_fileid = set([f[:-4] for f in os.listdir(video_folder)])
    extracted_fileid = set([f[:-4] for f in os.listdir(landmark_folder)])

    all_files = []
    for fid in all_fileid:
        if fid not in extracted_fileid:
            all_files.append(fid + ".mp4")

    print(f"{len(all_files)} are waiting for extracted")

    # process_video_files(all_files, video_folder, landmark_folder, validity_folder)

    # n = len(all_files) // num_cpus + 1
    # file_chuncks = [all_files[i:i + n] for i in range(0, len(all_files), n)]

    num_cpus = 10
    # split_idx = [int(len(robot_list) / num_cpus) * i for i in range(num_cpus)] + [len(robot_list)]
    split_idx = [0]
    len_list = len(all_files)
    num_parts = num_cpus
    while len_list > 0:
        sub_len = int(len_list / num_parts)
        split_idx.append(split_idx[-1] + sub_len)
        len_list -= sub_len
        num_parts -= 1

    process_args = []
    for i in range(num_cpus):
        file_chunk = all_files[split_idx[i]:split_idx[i + 1]]
        process_args.append((file_chunk, video_folder, landmark_folder, validity_folder))

    pool = Pool(processes=num_cpus)
    pool.starmap(process_video_files, process_args)

    print('Main finished')


if __name__ == '__main__':
    main()
