import datetime
import glob
import os
from typing import Tuple, List
# import sys

import cv2
import librosa
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import numpy as np

# from torch import from_numpy
from tqdm import tqdm
import scienceplots

from ambisonics.aem import AEMGenerator
# from ambisonics.distance import SphericalAmbisonicsVisualizer
# from ambisonics.aem import spherical_mesh


def compute_aem(
    audio_path: str,
    save_path: str = None,
    audio_format: str = "wav",
    duration: float = None,
    fps: int = 20,
    audio_frame_length: int = 4800,
    aem_width: int = 360,
    aem_height: int = 180,
    gpu: bool = False,
    batch_size: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute AEM for one audio file.
    """
    # Determine hop length
    _, sr = librosa.load(audio_path, mono=False, sr=None, duration=1)
    if sr % fps != 0:
        raise ValueError(f"Sample rate ({sr}) must be divisible by fps ({fps}).")
    audio_hop_length = sr // fps
    if audio_hop_length > audio_frame_length:
        raise Warning(
            f"Audio hop length ({audio_hop_length}) is larger than audio frame length ({audio_frame_length}), information might be lost."
        )

    if gpu:
        print("=> Using GPU for AEM computation.")

    aemg = AEMGenerator(
        audio_frame_length,
        audio_hop_length,
        n_phi=aem_width,
        n_nu=aem_height,
        gpu=gpu,
        batch_size=batch_size,
        show_progress=True,
    )

    y, sr_ = librosa.load(audio_path, mono=False, sr=None, duration=duration)
    if sr_ != sr:
        raise ValueError(f"Audio files have different sample rates: {sr_} and {sr}.")

    # compute aem
    # aem shape: (n_frames, n_phi, n_nu)
    aem = aemg.compute(y.T)

    # get time stamp
    time_stamp = np.arange(0, y.shape[1] - audio_frame_length, audio_hop_length) / sr_

    # save aem
    if save_path is not None:
        np.savez(
            save_path,
            time_stamp=time_stamp,
            phi_mesh=aemg.phi_mesh,
            nu_mesh=aemg.nu_mesh,
            aem=aem,
        )

    print(f"=> Done! AEMs saved to {save_path}.")

    return time_stamp, aemg.phi_mesh, aemg.nu_mesh, aem


def compute_aems(
    audio_dir: str,
    save_dir: str,
    audio_format: str = "wav",
    duration: float = None,
    fps: int = 20,
    audio_frame_length: int = 4800,
    aem_width: int = 360,
    aem_height: int = 180,
    gpu: bool = False,
    batch_size: int = 10,
):
    """
    Compure aems for all audio files in the audio_dir and save them in the save_dir.

    Args:
        audio_dir: str, path to the audio directory
        save_dir: str, path to the save directory
        duration: float, duration of the audio in seconds
        fps: int, frames per second
        audio_frame_length: int, frame length for audio in samples
        aem_width: int, width of the AEM
        aem_height: int, height of the AEM
        save_path: str, path to save the figure
        fig_size: tuple of int, figure size
        gpu: bool, use GPU for AEM computation
        batch_size: int, batch size for AEM computation
    """

    audio_files = glob.glob(f"{audio_dir}/*.{audio_format}")
    print(f"=> Found {len(audio_files)} audio files in {audio_dir}.")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # else:
    # raise ValueError(f"Existed save_dir: {save_dir}.")

    # Determine hop length
    _, sr = librosa.load(audio_files[0], mono=False, sr=None, duration=1)
    if sr % fps != 0:
        raise ValueError(f"Sample rate ({sr}) must be divisible by fps ({fps}).")
    audio_hop_length = sr // fps
    if audio_hop_length > audio_frame_length:
        raise Warning(
            f"Audio hop length ({audio_hop_length}) is larger than audio frame length ({audio_frame_length}), information might be lost."
        )

    if gpu:
        print("=> Using GPU for AEM computation.")

    aemg = AEMGenerator(
        audio_frame_length,
        audio_hop_length,
        n_phi=aem_width,
        n_nu=aem_height,
        gpu=gpu,
        batch_size=batch_size,
        show_progress=True,
    )

    audio_files_ = tqdm(audio_files, desc="=> Computing AEMs")
    for audio_path in audio_files_:
        audio_files_.set_description(f"=> Computing AEM for {audio_path}")

        y, sr_ = librosa.load(audio_path, mono=False, sr=None, duration=duration)
        if sr_ != sr:
            raise ValueError(
                f"Audio files have different sample rates: {sr_} and {sr}."
            )

        # compute aem
        # aem shape: (n_frames, n_phi, n_nu)
        aem = aemg.compute(y.T)

        # get time stamp
        time_stamp = (
            np.arange(0, y.shape[1] - audio_frame_length, audio_hop_length) / sr_
        )

        # save aem
        aem_name = os.path.basename(audio_path).replace(f".{audio_format}", ".npz")
        np.savez(
            os.path.join(save_dir, aem_name),
            time_stamp=time_stamp,
            phi_mesh=aemg.phi_mesh,
            nu_mesh=aemg.nu_mesh,
            aem=aem,
        )

    print(f"=> Done! AEMs saved to {save_dir}.")


def compute_anglegram_from_aem(aem_path):
    data = np.load(aem_path)
    time_stamp, phi_mesh, nu_mesh, aem = (
        data["time_stamp"],
        data["phi_mesh"],
        data["nu_mesh"],
        data["aem"],
    )
    phi_record, nu_record, energy_record = [], [], []

    aem_argmax = [np.unravel_index(np.argmax(r), r.shape) for r in aem]
    for i, point in enumerate(aem_argmax):
        phi_record.append(phi_mesh[point[0], point[1]])
        nu_record.append(nu_mesh[point[0], point[1]])
        energy_record.append(aem[i, point[0], point[1]])
    phi_record = np.array(phi_record)
    nu_record = np.array(nu_record)
    energy_record = np.array(energy_record)

    return time_stamp, np.stack([phi_record, nu_record]).T, energy_record


def get_anglegram_mean_rms(audio_path) -> Tuple[float, np.ndarray]:
    """
    Calculate average rms of the audio file, made for noise estimation..
    """
    time_stamp, phi_mesh, nu_mesh, aem = compute_aem(
        audio_path=audio_path,
        fps=20,
        audio_frame_length=4800,
        aem_width=360,  # angular_res = 1
        aem_height=180,
        gpu=True,
        batch_size=5,
    )
    phi_record, nu_record, energy_record = [], [], []

    aem_argmax = [np.unravel_index(np.argmax(r), r.shape) for r in aem]
    for i, point in enumerate(aem_argmax):
        phi_record.append(phi_mesh[point[0], point[1]])
        nu_record.append(nu_mesh[point[0], point[1]])
        energy_record.append(aem[i, point[0], point[1]])
    phi_record = np.array(phi_record)
    nu_record = np.array(nu_record)
    energy_record = np.array(energy_record)

    return np.mean(energy_record), energy_record


def reconstruct_array2_location(
    aem_paths: List[str],
    mic_positions: List[Tuple[float, float, float]],
):
    """
    Reconstruct location information from four mics in array2

    Parameters
    ----------
    aem_paths : List[str]
        path to the AEM files
    mic_positions : List[Tuple[float, float, float]]
        positions of the four microphones in the array in cm, order: x, y, z

    Returns
    -------
    timestamp: np.ndarray
    position: np.ndarray (n_frames, 3)
    """


def plot_anglegram(
    time_record,
    phi_record,
    nu_record,
    rms_record,
    db_threshold=-60,
    db_max=-20,
    title=None,
    save_path=None,
):
    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(2, 1, figsize=(12, 6))
        if title:
            fig.suptitle(title, fontsize=16)

        sc = ax[0].scatter(
            time_record[rms_record > db_threshold],
            phi_record[rms_record > db_threshold],
            c=rms_record[rms_record > db_threshold],
            label="Azimuth",
            alpha=0.9,
            vmin=db_threshold,
            vmax=db_max,
        )
        # find the closest multiple of 2pi to azimuth_min
        azimuth_min_pi = np.floor(np.min(phi_record) / np.pi) * np.pi
        azimuth_max_pi = np.ceil(np.max(phi_record) / np.pi) * np.pi
        ax[0].set_ylim(azimuth_min_pi - 0.1, azimuth_max_pi + 0.1)
        yticks = np.arange(azimuth_min_pi, azimuth_max_pi + 0.1, np.pi)
        ax[0].set_yticks(yticks)
        ax[0].set_yticklabels([f"{int(ytick / np.pi)}$\pi$" for ytick in yticks])

        # plt.colorbar(sc, ax=ax[0], label="RMS (dB)")
        ax[0].set(xlabel="Time (s)", ylabel="Azimuth (rad)")
        # horizontal lines on pi and 2pi
        for ytick in yticks:
            if np.abs(ytick % (2 * np.pi)) < 1e-1:
                ax[0].axhline(y=ytick, linestyle="-", color="black", alpha=0.5)
            else:
                ax[0].axhline(y=ytick, linestyle="--", alpha=0.3)

        ax[0].grid(axis="x")

        sc = ax[1].scatter(
            time_record[rms_record > db_threshold],
            nu_record[rms_record > db_threshold],
            c=rms_record[rms_record > db_threshold],
            label="Elevation",
            alpha=0.9,
            vmin=db_threshold,
            vmax=db_max,
        )
        # ax[1].plot(time_record, nu_record, label='Elevation')
        ax[1].set(
            xlabel="Time (s)",
            ylabel="Elevation (rad)",
            ylim=(-0.5 * np.pi - 0.1, 0.5 * np.pi + 0.1),
        )
        ax[1].grid(axis="x")
        yticks = np.array([-0.5 * np.pi, -0.25 * np.pi, 0, 0.25 * np.pi, 0.5 * np.pi])
        ax[1].set_yticks(yticks)
        ax[1].set_yticklabels(
            [r"$-\pi/2$", r"$-\pi/4$", r"$0$", r"$\pi/4$", r"$\pi/2$"]
        )
        for ytick in yticks:
            if ytick == 0:
                ax[1].axhline(y=ytick, linestyle="-", color="black", alpha=0.5)
            else:
                ax[1].axhline(y=ytick, linestyle="--", alpha=0.3)
        # plt.colorbar(sc, ax=ax[1], label="RMS (dB)")
        fig.colorbar(sc, ax=ax.ravel().tolist(), label="RMS (dB)")

    if save_path is not None:
        plt.savefig(save_path)
        print(f"=> Figure saved to {save_path}")
    else:
        plt.show()

    return (
        time_record[rms_record > db_threshold],
        phi_record[rms_record > db_threshold],
        nu_record[rms_record > db_threshold],
        rms_record[rms_record > db_threshold],
    )


def plot_anglemap(aem_path: str, save_path: str = None, **kwargs):
    data = np.load(aem_path)
    time_stamp, phi_mesh, nu_mesh, aem = (
        data["time_stamp"],
        data["phi_mesh"],
        data["nu_mesh"],
        data["aem"],
    )

    # print(f"=> aem shape: {aem.shape}") # (1093, 180, 360)

    phi_img = np.mean(aem, axis=1)  # (1093, 360)
    phi_img = np.flip(phi_img, axis=1).T  # (360, 1093), top left pi
    # print(f"=> phi_img shape: {phi_img.shape}")

    nu_img = np.mean(aem, axis=2)  # (1093, 180)
    nu_img = np.flip(nu_img, axis=1).T  # (180, 1093), top left pi/2
    # print(f"=> nu_img shape: {nu_img.shape}")

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(
        f"{os.path.basename(aem_path)}\nmin: {aem.mean():.2f}, max: {aem.max():.2f}",
        fontsize=16,
    )
    ax[0].imshow(phi_img, cmap="hot", aspect="auto", **kwargs)
    ax[0].set_yticks([0, 89, 179, 269, 359])
    ax[0].set_yticklabels([r"$\pi$", r"$\pi/2$", r"$0$", r"$-\pi/2$", r"$-\pi$"])
    ax[0].set_xticks(np.arange(0, len(time_stamp), 200))
    ax[0].set_xticklabels(np.arange(0, len(time_stamp), 200) / 20)

    ax[0].set_ylabel("Azimuth (rad)")

    ax[1].imshow(nu_img, cmap="hot", aspect="auto", **kwargs)
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Elevation (rad)")
    ax[1].set_yticks([0, 44, 89, 134, 179])
    ax[1].set_yticklabels([r"$\pi/2$", r"$\pi/4$", r"$0$", r"$-\pi/4$", r"$-\pi/2$"])

    if save_path is not None:
        plt.savefig(save_path)
        print(f"=> Figure saved to {save_path}")
    else:
        plt.show()


# def read_video_frames(video_path, skip_frames=3, frame_shape=(360, 180)):
#     """
#     Read video file into numpy array in shape (T, W, H, C)
#     Modified from baseline.
#     """

#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     frame_cnt = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if frame_cnt % skip_frames == 0:
#             if frame_shape is not None:
#                 frame = cv2.resize(frame, frame_shape)
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame_rgb)
#         frame_cnt += 1
#     cap.release()
#     cv2.destroyAllWindows()

#     return np.array(frames)


# def plot_sample_thumbnail(
#     sample_name: str,
#     save_path: str,
#     dataset_path: str = "/projects/ec12/DCASE24/dataset",
#     video_skip_frames: int = 3,
#     aem_skip_frames: int = 3,
#     n_rows: int = 4,
#     n_cols: int = 4,
#     fig_size: "tuple[int]" = (20, 20),
# ) -> None:
#     """
#     Plot a thumbnail of a sample with video frames and AEM.

#     Args:
#         sample_name: str, name of the sample
#         save_path: str, path to save the figure
#         video_skip_frames: int, skip frames for video
#         aem_skip_frames: int, skip frames for AEM
#     """
#     video_path = os.path.join(f"{dataset_path}/video_dev", sample_name + ".mp4")
#     aem_path = os.path.join(f"{dataset_path}/aem_dev", sample_name + ".npy")

#     # video shape: (n_frames, W, H, C), dtype: float32
#     video = (
#         read_video_frames(video_path, skip_frames=video_skip_frames).astype(np.float32)
#         / 255.0
#     )
#     # aem shape: (n_frames, n_phi(azimuth), n_nu(elevation)), dtype: float32
#     aem = np.load(aem_path)[::aem_skip_frames]
#     # normalize and flip aem
#     aem_min = np.expand_dims(aem.min(axis=-1).min(axis=-1), (1, 2))
#     aem_max = np.expand_dims(aem.max(axis=-1).max(axis=-1), (1, 2))
#     aem = (aem - aem_min) / (aem_max - aem_min)
#     aem = np.flip(aem, axis=2)

#     n_frames = min(len(video), len(aem))
#     frame_shape = (video.shape[2], video.shape[1])

#     # make cmap from transparent to red
#     c_white = mcolors.colorConverter.to_rgba("white", alpha=0)
#     c_red = mcolors.colorConverter.to_rgba("red", alpha=1)
#     cmap_rb = mcolors.LinearSegmentedColormap.from_list(
#         "rb_cmap", [c_white, c_red], 512
#     )

#     fig, ax = plt.subplots(n_rows, n_cols, squeeze=0, figsize=fig_size)
#     axi = ax.reshape(-1)
#     for i, frame_idx in enumerate(
#         np.linspace(0, n_frames - 1, n_rows * n_cols, dtype=int)
#     ):
#         axi[i].imshow(video[frame_idx])
#         axi[i].imshow(cv2.resize(aem[frame_idx], frame_shape), cmap=cmap_rb, alpha=0.5)
#         axi[i].set_title(frame_to_timestring(frame_idx, fps=29.97 / video_skip_frames))
#         axi[i].axis("off")
#     fig.suptitle(sample_name)

#     plt.tight_layout()
#     plt.savefig(save_path)
#     print(f"=> Figure saved to {save_path}")


# # compute_anglegram
# def compute_anglegram(
#     audio_file_path,
#     format,
#     time_res=0.2,
#     angular_res=2.0,
#     azimuth_offset=0.0,
#     start=0,
#     duration=10,
#     rms_threshold=-80,
#     unwrap_azimuth=False,
# ):
#     """
#     Plot the angle map of the audio file
#     :param audio_file_path: path to the audio file
#     :param time_res: time resolution in seconds
#     :param angular_res: angular resolution in radians
#     :param azimuth_offset: offset in radians
#     :param start: start time in seconds
#     :param duration: duration in seconds
#     :param rms_threshold: threshold in dB
#     """

#     assert format in ["AmbiX", "FuMa"]
#     y, sr = librosa.load(
#         audio_file_path, mono=False, sr=None, offset=start, duration=duration
#     )
#     viz = SphericalAmbisonicsVisualizer(
#         data=y.T, rate=sr, window=time_res, angular_res=angular_res
#     )
#     nu_mesh, phi_mesh = viz.mesh()  # nu is elevation, phi is azimuth
#     nu_mesh = nu_mesh[:, 0]
#     phi_mesh = phi_mesh[0, :]
#     nu_record, phi_record, time_record, rms_record = [], [], [], []
#     # print(phi_mesh)

#     # Get the maximum rms at each frame, and record its angle
#     rms = viz.get_next_frame()
#     time_record.append(0.0 + start)
#     nu_max, phi_max = np.unravel_index(np.argmax(rms), rms.shape)
#     nu_record.append(nu_mesh[nu_max])
#     phi_record.append(phi_mesh[phi_max])
#     rms_record.append(np.max(rms))

#     while True:
#         rms = viz.get_next_frame()
#         if rms is None:
#             break
#         time_record.append(time_record[-1] + time_res)
#         rms_record.append(np.max(rms))
#         # find the loudest point
#         nu_max, phi_max = np.unravel_index(np.argmax(rms), rms.shape)
#         nu_record.append(nu_mesh[nu_max])
#         phi_record.append(phi_mesh[phi_max])

#     # apply phase offset and unwrapping to azimuth
#     if unwrap_azimuth:
#         phi_record = np.unwrap(np.array(phi_record) + azimuth_offset, period=2 * np.pi)
#     else:
#         phi_record = np.array(phi_record) + azimuth_offset

#     rms_record = np.array(rms_record)
#     return (
#         np.array(time_record)[rms_record > rms_threshold],
#         np.array(phi_record)[rms_record > rms_threshold],
#         np.array(nu_record)[rms_record > rms_threshold],
#         librosa.amplitude_to_db(rms_record)[rms_record > rms_threshold],
#     )


# def plot_anglegram(
#     audio_file_path,
#     save_path=None,
#     db_threshold=-80,
#     # azimuth_line_start=0,
#     **kwargs,
# ):
#     # compute anglegram
#     time_record, phi_record, nu_record, rms_record = compute_anglegram(
#         audio_file_path,
#         **kwargs,
#     )

#     with plt.style.context(["science", "no-latex"]):
#         fig, ax = plt.subplots(2, 1, figsize=(12, 6))
#         fig.suptitle(os.path.basename(audio_file_path), fontsize=16)

#         sc = ax[0].scatter(
#             time_record[rms_record > db_threshold],
#             phi_record[rms_record > db_threshold],
#             c=rms_record[rms_record > db_threshold],
#             label="Azimuth",
#             alpha=0.9,
#         )
#         # find the closest multiple of 2pi to azimuth_min
#         azimuth_min_2pi = np.floor(np.min(phi_record) / (2 * np.pi)) * 2 * np.pi
#         azimuth_max_2pi = np.ceil(np.max(phi_record) / (2 * np.pi)) * 2 * np.pi
#         ax[0].set_ylim(azimuth_min_2pi - 0.1, azimuth_max_2pi + 0.1)
#         yticks = np.arange(azimuth_min_2pi, azimuth_max_2pi + 0.1, np.pi)
#         ax[0].set_yticks(yticks)
#         ax[0].set_yticklabels([f"{int(ytick / np.pi)}$\pi$" for ytick in yticks])

#         plt.colorbar(sc, ax=ax[0], label="RMS (dB)")
#         ax[0].set(xlabel="Time (s)", ylabel="Azimuth (rad)")
#         # horizontal lines on pi and 2pi
#         for ytick in yticks:
#             if np.abs(ytick % (2 * np.pi)) < 1e-1:
#                 ax[0].axhline(y=ytick, linestyle="-", color="black", alpha=0.5)
#             else:
#                 ax[0].axhline(y=ytick, linestyle="--", alpha=0.3)

#         ax[0].grid(axis="x")

#         sc = ax[1].scatter(
#             time_record[rms_record > db_threshold],
#             nu_record[rms_record > db_threshold],
#             c=rms_record[rms_record > db_threshold],
#             label="Elevation",
#             alpha=0.9,
#         )
#         # ax[1].plot(time_record, nu_record, label='Elevation')
#         ax[1].set(
#             xlabel="Time (s)",
#             ylabel="Elevation (rad)",
#             ylim=(-0.5 * np.pi - 0.1, 0.5 * np.pi + 0.1),
#         )
#         ax[1].grid(axis="x")
#         yticks = np.array([-0.5 * np.pi, -0.25 * np.pi, 0, 0.25 * np.pi, 0.5 * np.pi])
#         ax[1].set_yticks(yticks)
#         ax[1].set_yticklabels(
#             [r"$-\pi/2$", r"$-\pi/4$", r"$0$", r"$\pi/4$", r"$\pi/2$"]
#         )
#         for ytick in yticks:
#             if ytick == 0:
#                 ax[1].axhline(y=ytick, linestyle="-", color="black", alpha=0.5)
#             else:
#                 ax[1].axhline(y=ytick, linestyle="--", alpha=0.3)
#         plt.colorbar(sc, ax=ax[1], label="RMS (dB)")

#     if save_path is not None:
#         plt.savefig(save_path)
#         print(f"=> Figure saved to {save_path}")
#     else:
#         plt.show()

#     return (
#         time_record[rms_record > db_threshold],
#         phi_record[rms_record > db_threshold],
#         nu_record[rms_record > db_threshold],
#         rms_record[rms_record > db_threshold],
#     )


def frame_to_seconds(frame_idx, fps=30):
    return frame_idx / fps


def seconds_to_timestring(seconds):
    return str(datetime.timedelta(seconds=seconds))


def frame_to_timestring(frame_idx, fps=30):
    return seconds_to_timestring(frame_to_seconds(frame_idx, fps))


if __name__ == "__main__":
    ## compute all aems
    compute_aems(
        audio_dir='../audio/trimmed/Bformat/',
        save_dir='../audio/trimmed/AEM',
        fps=20,
        audio_frame_length=4800,
        aem_width=360, # angular_res = 1
        aem_height=180,
        gpu=True,
        batch_size=5
    )

    ## plot all anglegrams
    # path = "/fp/homes01/u01/ec-jinyueg/felles_/Research/Project/AMBIENT/RoomAmbisonicsPaper"
    # aems = glob.glob(f"{path}/audio/trimmed/AEM/*.npz")
    # for sample in tqdm(aems):
    #     plot_anglegram(
    #         *compute_anglegram_from_aem(sample),
    #         db_threshold=-40,
    #         title=os.path.basename(sample),
    #         save_path=f"{path}/figures/anglegrams_uniform_db/{os.path.basename(sample).replace('.npz', '.png')}",
    #     )
