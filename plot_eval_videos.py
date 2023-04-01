import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import subprocess
import soundfile as sf
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Process
from utils.utils import mash_to_contour, face_3d_to_2d, contour_connections


def draw_lm(pred_lm, gt_lm, save_path):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    gt = mash_to_contour(face_3d_to_2d(gt_lm))
    axs[0].scatter(gt[0], gt[1], s=3, c='tab:blue')
    axs[2].scatter(gt[0], gt[1], s=3, c='tab:blue')
    for c in contour_connections:
        point_1 = gt[:, c[0]]
        point_2 = gt[:, c[1]]
        axs[0].plot([point_1[0], point_2[0]], [point_1[1], point_2[1]], c='tab:blue', zorder=0)
        axs[2].plot([point_1[0], point_2[0]], [point_1[1], point_2[1]], c='tab:blue', zorder=0)

    pred = mash_to_contour(face_3d_to_2d(pred_lm))
    axs[1].scatter(pred[0], pred[1], s=3, c='tab:orange')
    axs[2].scatter(pred[0], pred[1], s=3, c='tab:orange')
    for c in contour_connections:
        point_1 = pred[:, c[0]]
        point_2 = pred[:, c[1]]
        axs[1].plot([point_1[0], point_2[0]], [point_1[1], point_2[1]], c='tab:orange', zorder=0)
        axs[2].plot([point_1[0], point_2[0]], [point_1[1], point_2[1]], c='tab:orange', zorder=0)

    axs[0].set_title("GT")
    axs[1].set_title("PRED")
    axs[2].set_title("MIX")

    for ax in axs:
        ax.set_axis_off()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(save_path)
    plt.close()


def process_files(args):
    p_idx, rank_idx_chunk, save_dir, pred_lms, gt_lms, mses, raw_audios = args

    tmp_vid_dir = os.path.join(save_dir, f"tmp_vid_{p_idx}")
    os.makedirs(tmp_vid_dir, exist_ok=True)
    for rank, idx in tqdm(rank_idx_chunk):
        pred_lm = pred_lms[idx]
        gt_lm = gt_lms[idx]
        mse = mses[idx]
        raw_audio = raw_audios[idx].flatten()

        for i in range(125):
            pred_lm_frame = pred_lm[i]
            gt_lm_frame = gt_lm[i]
            draw_lm(pred_lm_frame, gt_lm_frame,
                    save_path=os.path.join(tmp_vid_dir, f"{i:04d}.png"))

        tmp_wav_path = os.path.join(save_dir, f'tmp_{p_idx}.wav')
        tmp_vid_path = os.path.join(save_dir, f'tmp_{p_idx}.mp4')
        output_vid = os.path.join(save_dir, f"rank_{rank:04d}_idx_{idx:04d}_mse{mse:.4f}.mp4")
        cmd = f"ffmpeg -r 25 -f image2 -i {tmp_vid_dir}\\%04d.png -vcodec libx264 -pix_fmt yuv420p {tmp_vid_path}"
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        p.wait()

        sf.write(tmp_wav_path, raw_audio, 16000)

        cmd = f"ffmpeg -i {tmp_vid_path} -i {tmp_wav_path} -c:v copy -c:a aac {output_vid}"
        p = subprocess.Popen(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        p.wait()

        for file in os.listdir(tmp_vid_dir):
            file_path = os.path.join(tmp_vid_dir, file)
            os.remove(file_path)
        os.remove(tmp_wav_path)
        os.remove(tmp_vid_path)

if __name__ == '__main__':

    save_dir = "eval_videos"
    os.makedirs(save_dir, exist_ok=True)

    compare_data = pickle.load(open("predlm_gtlm_mse.pkl", 'rb'))
    pred_lms, gt_lms, mses, raw_audios = compare_data

    err_idx = np.argsort(mses)

    k = 13
    rank_idx = [(rank, idx) for rank, idx in enumerate(err_idx)]
    chunk_size = len(rank_idx) // k
    rank_idx_chunk = [rank_idx[i:i + chunk_size] for i in range(0, len(rank_idx), chunk_size)]
    rank_idx_chunk = list(enumerate(rank_idx_chunk))

    procs = []
    for idx, rank_idx in rank_idx_chunk:
        proc = Process(target=process_files, args=((idx, rank_idx, save_dir, pred_lms, gt_lms, mses, raw_audios),))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()