import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import *
from librosa.effects import split

landmarks = []
validity = []
audio_chunks = []
for fid in tqdm(os.listdir('..\\data\\landmarks')[:2]):
    lms = np.load(os.path.join('..\\data\\landmarks', fid))
    valid = np.load(os.path.join('..\\data\\validity', fid))
    audio = np.load(os.path.join('..\\data\\audios', fid))
    audio_pad = np.hstack((audio, np.zeros(16000)))
    audio_trim = audio_pad[:int(lms.shape[0] * 16000 / 25)]
    audio_chunk = np.vstack(np.split(audio_trim, valid.shape[0]))
    landmarks.append(lms)
    validity.append(valid)
    audio_chunks.append(audio_chunk)

landmarks = np.vstack(landmarks)
validity = np.hstack(validity)
audio_chunks = np.vstack(audio_chunks)


def get_mean_face(landmarks, validity, audio_chunks):
    mean_face = landmarks[validity].mean(axis=0)
    return mean_face

def get_close_mouth(landmarks, validity, audio_chunks):
    landmark_contour = mash_to_contour(face_3d_to_2d(landmarks[validity], seq=True), seq=True)
    inner_lips_area = np.array([get_inner_lip_area(lc) for lc in landmark_contour])
    area_idx = np.argsort(inner_lips_area)
    close_mouth_lm = landmarks[validity][area_idx[:250]]
    close_mouth_face = close_mouth_lm.mean(axis=0)
    return close_mouth_face

audio_wav = audio_chunks.flatten()
s = split(audio_wav, top_db=20)

abs_aud = np.abs(audio_chunks).mean(axis=1)
# plt.hist(abs_aud[validity & (abs_aud < 0.001)], bins=300)
# plt.show()


print(len(landmarks[validity & (abs_aud < 0.0025)]))
slience_mean_face = landmarks[validity & (abs_aud < 0.0025)].mean(axis=0)
# plt.plot(abs_aud)
plt.plot(audio_wav, c='red')
for i, j in s:
    plt.plot(range(i, j), audio_wav[i:j], c='blue')

plt.show()
np.save('mean_face', mean_face)
# np.save('slient_mean_face_0.0025', slience_mean_face)
#
# np.save('close_mouth_face', close_mouth_face)