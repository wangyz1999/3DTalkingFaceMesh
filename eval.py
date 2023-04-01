import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import WhisperModel, Wav2Vec2Model

from dataset import get_dataloaders
from model import Faceformer
from utils.drawing_utils import lips_idx, inner_lips_idx

lips_lm = lips_idx + inner_lips_idx


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_best_model(model_folder):
    models = os.listdir(model_folder)
    model_epochs = [(model_name, float(model_name.split('valid')[1].split('.model')[0])) for model_name in models]
    model_epochs.sort(key=lambda x: x[1])
    best_model = model_epochs[0][0]
    print("Loaded Model:", best_model)
    return os.path.join(model_folder, best_model)


def test(run_dir):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(os.path.join(run_dir, 'config.json'), 'r') as f:
        args.__dict__ = json.load(f)

    args.worker = 2
    args.batch_size = 8
    args.repeat = 'repeat'

    dataset = get_dataloaders(args)

    # build model
    model = Faceformer(args)
    model.eval()
    print("model parameters: ", count_parameters(model))

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(torch.device("cuda"))
    best_model_path = find_best_model(args.model_dir)
    model.load_state_dict(torch.load(best_model_path))

    if args.audio_encoder == 'whisper':
        audio_encoder = WhisperModel.from_pretrained("openai/whisper-small").encoder.cuda()
        audio_encoder._freeze_parameters()

    elif args.audio_encoder == 'wav2vec':
        audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").cuda()
        audio_encoder.freeze_feature_encoder()

    face_mses = []
    lips_mses = []
    face_vel_mses = []
    lips_vel_mses = []
    for batch in tqdm(dataset['test']):
        audio, displace = batch
        audio, displace = audio.to(device="cuda"), displace.to(device="cuda")

        with torch.no_grad():
            audio_embedding = audio_encoder(audio).last_hidden_state
            frame_num = displace.shape[1]
            displace_out = model.inference(audio_embedding, frame_num).reshape(args.batch_size, -1, 3, 468)
            displace_vel = torch.diff(displace, dim=1)
            displace_out_vel = torch.diff(displace_out, dim=1)

            face_mse = F.mse_loss(displace.flatten(start_dim=1), displace_out.flatten(start_dim=1),
                                  reduction='none').mean(dim=1).detach().cpu().numpy()
            lips_mse = F.mse_loss(displace[..., lips_lm].flatten(start_dim=1),
                                  displace_out[..., lips_lm].flatten(start_dim=1), reduction='none').mean(
                dim=1).detach().cpu().numpy()
            face_vel_mse = F.mse_loss(displace_vel.flatten(start_dim=1), displace_out_vel.flatten(start_dim=1),
                                      reduction='none').mean(dim=1).detach().cpu().numpy()
            lips_vel_mse = F.mse_loss(displace_vel[..., lips_lm].flatten(start_dim=1),
                                      displace_out_vel[..., lips_lm].flatten(start_dim=1), reduction='none').mean(
                dim=1).detach().cpu().numpy()

            face_mses.append(face_mse)
            lips_mses.append(lips_mse)
            face_vel_mses.append(face_vel_mse)
            lips_vel_mses.append(lips_vel_mse)

    face_mses = np.hstack(face_mses)
    lips_mses = np.hstack(lips_mses)
    face_vel_mses = np.hstack(face_vel_mses)
    lips_vel_mses = np.hstack(lips_vel_mses)

    face_mse_mean = face_mses.mean()
    face_mse_std = face_mses.std()
    lips_mse_mean = lips_mses.mean()
    lips_mse_std = lips_mses.std()

    face_vel_mse_mean = face_vel_mses.mean()
    face_vel_mse_std = face_vel_mses.std()
    lips_vel_mse_mean = lips_vel_mses.mean()
    lips_vel_mse_std = lips_vel_mses.std()

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
    return result


def main():
    log_dir = "log\\encoder_audio_window_2"
    eval_file = open(os.path.join(log_dir, 'eval.csv'), 'w', newline='')
    eval_csv = csv.writer(eval_file)
    eval_csv.writerow(['', 'face_err', 'face_err_std', 'lips_err', 'lips_err_std', 'face_vel_err', 'face_vel_err_std', 'lips_vel_err', 'lips_vel_err_std'])
    for run_id in os.listdir(log_dir):
        if 'run' not in run_id:
            continue
        run_dir = os.path.join(log_dir, run_id)
        result = test(run_dir)
        eval_csv.writerow([
            run_id,
            result['face_mean'],
            result['face_std'],
            result['lips_mean'],
            result['lips_std'],
            result['face_vel_mean'],
            result['face_vel_std'],
            result['lips_vel_mean'],
            result['lips_vel_std'],
        ])
    eval_file.close()


if __name__ == "__main__":
    main()
