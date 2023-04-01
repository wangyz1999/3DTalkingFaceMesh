import argparse
import os, shutil
import subprocess
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from tqdm import tqdm

from model import Audio2Landmark
from transformers import WhisperModel, AutoFeatureExtractor, Wav2Vec2Processor, Wav2Vec2Model
from utils.drawing_utils import save_landmark_video


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def find_best_model(model_folder):
    models = os.listdir(model_folder)
    model_epochs = [(model_name, float(model_name.split('valid')[1].split('.model')[0])) for model_name in models]
    model_epochs.sort(key=lambda x: x[1])
    best_model = model_epochs[0][0]
    print("Loaded Model:", best_model)
    return os.path.join(model_folder, best_model)


def generate_mosaic_command(video_paths, w, h, output_path):
    # Determine the number of tiles needed
    num_tiles = w * h
    num_videos = len(video_paths)
    if num_videos != num_tiles:
        raise ValueError(f"Not enough videos to fill mosaic of size {w}x{h}")

    # Generate the FFmpeg command string
    cmd = f"ffmpeg -loglevel warning -y"
    for i in range(num_tiles):
        cmd += f" -i {video_paths[i]}"
    cmd += " -filter_complex "
    cmd += f"xstack=grid={w}x{h}"
    cmd += f" {output_path}"
    return cmd

def smooth_landmarks(faces, w):
    """
    Compute moving average with width w along the time dimension
    :param faces: facial landmarks, time x dimension x num_of_keypoints
    :param w: width of the moving average
    :return: smoothed landmarks
    """
    faces = np.concatenate((np.repeat(np.expand_dims(faces[0], 0), w - 1, axis=0), faces), axis=0)
    ret = np.cumsum(faces, axis=0)
    ret[w:] = ret[w:] - ret[:-w]
    return ret[w - 1:] / w


def exp_smooth_landmarks(faces, alpha=0.65):
    faces = np.concatenate((np.repeat(np.expand_dims(faces[0], 0), 2 - 1, axis=0), faces), axis=0)
    sm_faces = []
    for i in range(2, faces.shape[0]):
        sm_faces.append(alpha * faces[i] + (1-alpha) * faces[i-1])
        # sm_faces.append(0.45 * faces[i] + 0.35 * faces[i - 1] + 0.2 * faces[i-2])
    sm_faces = np.stack(sm_faces)
    return sm_faces

def create_demo(run_dir):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(os.path.join(run_dir, 'config.json'), 'r') as f:
        args.__dict__ = json.load(f)

    args.demo_audio_path = r"data\demo_audio_chatgpt"
    # build model
    model = Audio2Landmark(args)
    print("model parameters: ", count_parameters(model))

    # to cuda
    assert torch.cuda.is_available()
    model.eval()
    model = model.to(torch.device("cuda"))
    best_model_path = find_best_model(args.model_dir)
    model.load_state_dict(torch.load(best_model_path))
    ref_face = np.load(args.ref_face_path)

    if args.audio_encoder == 'whisper':
        audio_encoder = WhisperModel.from_pretrained("openai/whisper-small").encoder.cuda()
        audio_encoder._freeze_parameters()
        feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-small")
    elif args.audio_encoder == 'wav2vec':
        audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").cuda()
        audio_encoder.freeze_feature_encoder()
        feature_extractor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    video_folder = os.path.join(args.run_dir, "video")
    video_audio_folder = os.path.join(args.run_dir, "video_audio")
    landmark_folder = os.path.join(args.run_dir, "landmark")
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(video_audio_folder, exist_ok=True)
    os.makedirs(landmark_folder, exist_ok=True)

    for audio in os.listdir(args.demo_audio_path):
        print("processing:", audio)
        file_id = audio[:-4]
        audio_path = os.path.join(args.demo_audio_path, audio)
        video_path = os.path.join(video_folder, file_id + ".mp4")
        lm_path = os.path.join(landmark_folder, file_id)
        video_audio_path = os.path.join(video_audio_folder, file_id + ".mp4")
        y, sr = librosa.load(audio_path, sr=16000)
        step = args.audio_len
        landmarks = []
        for i in tqdm(range(0, len(y), step * 16000)):
            y_chunk = y[i:i + step * 16000].copy()
            y_chunk.resize(step * 16000)
            if args.audio_encoder == 'whisper':
                audio_mel_spec = feature_extractor(y_chunk, sampling_rate=16000, return_tensors="pt").input_features.cuda()
                audio_embedding = audio_encoder(audio_mel_spec).last_hidden_state
            elif args.audio_encoder == 'wav2vec':
                input_values = feature_extractor(y_chunk, sampling_rate=16000, return_tensors="pt").input_values.cuda()
                audio_embedding = audio_encoder(input_values).last_hidden_state

            frame_num = step * 25
            with torch.no_grad():
                displace = model.inference(audio_embedding, frame_num).detach().cpu().numpy()

            pred_landmark = ref_face + displace[0].reshape(-1, 3, 468)
            landmarks.append(pred_landmark)
            # np.save(audio[:-4], pred_landmark)
        landmarks = np.vstack(landmarks)

        landmarks = smooth_landmarks(landmarks, 2)

        np.save(lm_path, landmarks)
        textt = "close mouth" if args.ref_face_path == "data\close_mouth_face.npy" else "open mouth"
        save_landmark_video(landmarks, video_path, text=f"{textt}")

        process = subprocess.Popen(['ffmpeg',
                                    '-i', video_path,
                                    '-i', audio_path,
                                    '-c:a', 'aac',
                                    '-y',
                                    '-loglevel', 'warning',
                                    video_audio_path])
    process.wait()
    shutil.rmtree(video_folder)

def create_tile_demo(log_dir, wid, hei):
    demo_folder = "demo_tile"
    video_list = os.listdir(os.path.join(log_dir, 'run_0', 'video_audio'))
    tile_video_path = os.path.join(log_dir, demo_folder)
    os.makedirs(tile_video_path, exist_ok=True)
    processes = []
    for video in video_list:
        videos = []
        for run_id in os.listdir(log_dir):
            if run_id == demo_folder:
                continue
            run_dir = os.path.join(log_dir, run_id)
            videos.append(os.path.join(run_dir, f'video_audio\\{video}'))
        output_path = os.path.join(tile_video_path, video)
        cmd = generate_mosaic_command(videos, w=wid, h=hei, output_path=output_path)
        process = subprocess.Popen(cmd.split())
        processes.append(process)
    for p in processes:
        p.wait()

def main():
    log_dir = "log\\open_close_mouth"
    wid = 2
    hei = 1
    for run_id in os.listdir(log_dir):
        if 'run' not in run_id:
            continue
        run_dir = os.path.join(log_dir, run_id)
        create_demo(run_dir)
    create_tile_demo(log_dir, wid, hei)


if __name__ == "__main__":
    main()
