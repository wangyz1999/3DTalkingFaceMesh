import argparse
import os
import random
import json

import numpy as np
import torch

from dataset import get_dataloaders
from model import Audio2Landmark
from trainer import Trainer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_experiment(args):
    # seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # make experiment directory
    os.makedirs(args.log_path, exist_ok=True)
    run_dir = os.path.join(args.log_path, f"run_{len(os.listdir(args.log_path))}")
    os.makedirs(run_dir, exist_ok=False)
    args.run_dir = run_dir
    model_dir = os.path.join(run_dir, "models")
    os.makedirs(model_dir, exist_ok=False)
    args.model_dir = model_dir

    # log config
    with open(f"{run_dir}\\config.json", 'w') as f:
        json.dump(args.__dict__, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Speech-driven Talking 3D Face Mesh using Transformer')
    parser.add_argument("--seed", type=int, default=44, help='seed of experiment')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--audio_path", type=str, default=r"data\audios", help='path of audio data')
    parser.add_argument("--demo_audio_path", type=str, default=r"data\demo_audio", help='path of audio data')
    parser.add_argument("--landmark_path", type=str, default=r"data\landmarks", help='path of landmark data')
    parser.add_argument("--validity_path", type=str, default=r"data\validity", help='path of validity data')
    parser.add_argument("--ref_face_path", type=str, default=r"data\close_mouth_face.npy", help='path of validity data')
    parser.add_argument("--data_split_path", type=str, default=r"data\train_test_split.txt", help='path of data split')
    parser.add_argument("--log_path", type=str, default=r"log\open_close_mouth", help='path of experiment logging')
    parser.add_argument("--vertice_dim", type=int, default=468 * 3, help='number of vertices of openpose')
    parser.add_argument("--feature_dim", type=int, default=64, help='hidden dim of transformer input')
    parser.add_argument("--align_window", type=int, default=2, help='width of audio alignment window attention mask')
    parser.add_argument("--period", type=int, default=25, help='period in Perodic Positional Encoding')
    parser.add_argument("--max_iter", type=int, default=5000, help='number of iterations')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--optimizer", type=str, default='lion', help='optimizer: adam, adamw, lion')
    parser.add_argument("--batch_size", type=int, default=8, help='batch size')
    parser.add_argument("--worker", type=int, default=2, help='data loader num_worker')
    parser.add_argument("--repeat", type=str, default='repeat', help='repeat_type, repeat vs interleave')
    parser.add_argument("--audio_len", type=int, default=5, help='length of audio input in secs')
    parser.add_argument("--output_type", type=str, default='disp_close', help='pos, disp_close, disp_mean')
    parser.add_argument("--audio_encoder", type=str, default="whisper", help='whisper or wav2vec or apc')
    parser.add_argument("--loss_wid", type=int, default=50, help='window length of moving average training loss')
    parser.add_argument("--log_interval", type=int, default=10, help='tensorboard training log interval')
    parser.add_argument("--valid_interval", type=int, default=100, help='do validation every this many train iters')
    parser.add_argument("--valid_steps", type=int, default=20, help='each validation takes this many iters')
    args = parser.parse_args()

    init_experiment(args)
    dataset = get_dataloaders(args)

    # build model
    model = Audio2Landmark(args)
    model = model.to(torch.device("cuda"))
    print("model parameters: ", count_parameters(model))

    # Train the model
    trainer = Trainer(args, model, dataset)
    trainer.train()


if __name__ == "__main__":
    main()
