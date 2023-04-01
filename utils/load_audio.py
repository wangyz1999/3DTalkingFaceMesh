import librosa
import os
from tqdm import tqdm
import numpy as np
from multiprocessing import Process, cpu_count

def load_audio(datalist):
    for vid in tqdm(datalist):
        yvid = vid[:-4]
        vid_pth = os.path.join('videos', vid)
        y, sr = librosa.load(vid_pth, sr=16000)
        np.save(os.path.join('audios', yvid), y)

def reduce_frame_rate(datalist):
    for vid in tqdm(datalist):
        yvid = vid[:-4]
        vid_pth = os.path.join('videos', vid)
        out_vid_pth = os.path.join('videos_25', vid)
        down_video = " ".join([
            "ffmpeg",
            "-hwaccel", "cuda",
            '-i', vid_pth,
            '-filter:v',
            'fps=fps=25',
            '-loglevel', 'warning',
            out_vid_pth
        ])
        status = os.system(down_video)


if __name__ == '__main__':
    data_list = os.listdir('videos')
    # num_cpus = 1
    num_cpus = cpu_count()
    split_idx = [int(len(data_list) / num_cpus) * i for i in range(num_cpus)] + [len(data_list)]
    processes = []
    for i in range(num_cpus):
        process = Process(target=reduce_frame_rate, args=(data_list[split_idx[i]:split_idx[i + 1]],))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()