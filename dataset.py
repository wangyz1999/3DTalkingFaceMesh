import os
import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm
import random
from transformers import AutoFeatureExtractor, Wav2Vec2Processor

class Dataset(data.Dataset):
    def __init__(self, data, audio_len, encoder):
        self.data = data
        self.data_slices = []
        self.frame_len = audio_len * 25
        self.get_data_slices()
        self.encoder = encoder
        if encoder == 'whisper':
            self.feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-small")
        elif encoder == 'wav2vec':
            self.feature_extractor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    def __getitem__(self, index):
        l_idx, r_idx = self.data_slices[index]
        audio = self.data['audio'][l_idx:r_idx].astype("float32")
        displace = self.data['displace'][l_idx:r_idx].astype("float32")

        audio_wav = audio.flatten()
        if self.encoder == 'whisper':
            audio_feat = self.feature_extractor(audio_wav, sampling_rate=16000, return_tensors="pt").input_features[0]
        elif self.encoder == 'wav2vec':
            audio_feat = self.feature_extractor(audio_wav, sampling_rate=16000, return_tensors="pt").input_values[0]

        return audio_feat, displace, audio

    def __len__(self):
        return self.dataset_length

    def get_data_slices(self):
        l_idx = 0
        while True:
            self.data_slices.append((l_idx, l_idx+self.frame_len))
            l_idx += self.frame_len
            if l_idx + self.frame_len > self.data['audio'].shape[0]:
                break
        self.dataset_length = len(self.data_slices)
        print("Total Slices:", self.dataset_length)


def read_data(args, test=False):
    print("Loading data...")
    audio_path = args.audio_path
    vertices_path = args.landmark_path
    validity_path = args.validity_path
    ref_face = np.load(args.ref_face_path).astype("float32")

    train_audio, valid_audio, test_audio = [], [], []
    train_displace, valid_displace, test_displace = [], [], []
    file_id_type = [tuple(fid.split('\t')) for fid in open(args.data_split_path).read().split('\n')]

    for f, data_type in tqdm(file_id_type):
        if test and data_type != "test":
            continue
        speech_array = np.load(os.path.join(audio_path, f)).astype("float32")
        landmark_array = np.load(os.path.join(vertices_path, f)).astype("float32")
        validity_array = np.load(os.path.join(validity_path, f))

        # turn invalid audio frames to silence
        audio_pad = np.hstack((speech_array, np.zeros(16000)))
        audio_trim = audio_pad[:int(validity_array.shape[0] * 16000 / 25)]
        audio_chunk = np.vstack(np.split(audio_trim, validity_array.shape[0]))
        audio_chunk[~validity_array] = 0

        # turn invalid landmark frames to close mouth
        landmark_array[~validity_array] = ref_face

        displace = landmark_array - ref_face

        if data_type == "train":
            train_audio.append(audio_chunk)
            train_displace.append(displace)
        elif data_type == "valid":
            valid_audio.append(audio_chunk)
            valid_displace.append(displace)
        elif data_type == "test":
            test_audio.append(audio_chunk)
            test_displace.append(displace)
        else:
            raise Exception("Invalid Data Type")

    test_data = {
        "audio": np.vstack(test_audio),
        "displace": np.vstack(test_displace)
    }
    if test:
        return test_data

    train_data = {
        "audio": np.vstack(train_audio),
        "displace": np.vstack(train_displace)
    }
    valid_data = {
        "audio": np.vstack(valid_audio),
        "displace": np.vstack(valid_displace)
    }

    assert train_data["audio"].shape[0] == train_data["displace"].shape[0]
    assert valid_data["audio"].shape[0] == valid_data["displace"].shape[0]
    assert test_data["audio"].shape[0] == test_data["displace"].shape[0]

    return train_data, valid_data, test_data


def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data = read_data(args)
    train_data = Dataset(train_data, args.audio_len, args.audio_encoder)
    dataset["train"] = data.DataLoader(dataset=train_data,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=args.worker,
                                       pin_memory=True,
                                       persistent_workers=True)
    valid_data = Dataset(valid_data, args.audio_len, args.audio_encoder)
    dataset["valid"] = data.DataLoader(dataset=valid_data,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=args.worker,
                                       persistent_workers=True)
    test_data = Dataset(test_data, args.audio_len, args.audio_encoder)
    dataset["test"] = data.DataLoader(dataset=test_data,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.worker,
                                      drop_last=True)
    return dataset


def get_test_dataloaders(args):
    dataset = {}
    test_data = read_data(args, test=True)
    test_data = Dataset(test_data, args.audio_len, args.audio_encoder)
    dataset["test"] = data.DataLoader(dataset=test_data,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.worker,
                                      drop_last=True)
    return dataset

if __name__ == "__main__":
    get_dataloaders()