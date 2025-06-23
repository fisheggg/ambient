import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerNpoints
from tqdm import tqdm

import anglegram
import mocap

MICNAME_MAP = {
    "Mic1": "Mic B1",
    "Mic2": "Mic B2",
    "Mic3": "Mic B3",
    "Mic4": "Mic B4",
    "Mic5": "Mic A",
}


def compute_doae(anglegram_path, mocap_path, mic_name, anglegram_db_threshold=-40):
    """
    Compute Direction of Arrival Error (DOAE) between anglegram and mocap data.

    Returns
    -------
    timestamp : np.array (n_frames,)
    diff : np.array (n_frames, 2)
        modded abs difference between anglegram and mocap data in radians
        columns: azimuth and elevation
    anglegram_polar : np.array (n_frames, 2)
        anglegram data in radians
        columns: azimuth and elevation
    mocap_polar : np.array (n_frames, 3)
        mocap data in radians
        columns: azimuth, elevation, radius
    """
    # load mocap data (fps 240)
    mocap_polar, mocap_timestamp = mocap.get_direction_to_mic(
        mocap_path,
        mic_name,
        "polar",
    )
    # resample mocap data to 20 fps
    mocap_polar = mocap_polar[::12, :]
    mocap_timestamp = mocap_timestamp[::12]

    # load anglegram data (fps 20)
    anglegram_timestamp, anglegram_polar, anglegram_rms = (
        anglegram.compute_anglegram_from_aem(
            anglegram_path,
        )
    )
    # threshold anglegram
    for col in range(anglegram_polar.shape[1]):
        anglegram_polar[:, col] = np.where(
            anglegram_rms < anglegram_db_threshold,
            np.array([np.nan] * len(anglegram_polar)),
            anglegram_polar[:, col],
        )
    anglegram_rms = np.where(
        anglegram_rms < anglegram_db_threshold, np.nan, anglegram_rms
    )

    # trim if necessary
    length = np.min([len(mocap_timestamp), len(anglegram_timestamp)])
    # assert mocap_timestamp == anglegram_timestamp

    # calculate the difference, mod by 2pi
    array = np.stack(
        [
            anglegram_polar[:length, :] - mocap_polar[:length, :2],
            anglegram_polar[:length, :]
            - mocap_polar[:length, :2]
            + 2 * np.pi * np.ones_like(anglegram_polar[:length, :]),
            anglegram_polar[:length, :]
            - mocap_polar[:length, :2]
            - 2 * np.pi * np.ones_like(anglegram_polar[:length, :]),
        ],
        axis=0,
    )

    diff = np.min(np.abs(array), axis=0)

    return (
        anglegram_timestamp[:length],
        diff[:length],
        anglegram_polar[:length],
        anglegram_rms[:length],
        mocap_polar[:length],
    )


def compute_and_plot_doae(
    anglegram_path, mocap_path, mic_name, anglegram_db_threshold=-40, save_path=None
):
    """
    Compute and plot Direction of Arrival Error (DOAE) between anglegram and mocap data.

    """
    timestamp, diff, anglegram_polar, anglegram_rms, mocap_polar = compute_doae(
        anglegram_path,
        mocap_path,
        mic_name,
        anglegram_db_threshold=anglegram_db_threshold,
    )

    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, dpi=300)
        mocap_dots = ax[0].scatter(
            timestamp,
            mocap_polar[:, 0],
            label="Motion capture",
            alpha=0.8,
            s=10,
            color="tab:pink",
        )
        anglegram_dots = ax[0].scatter(
            timestamp,
            anglegram_polar[:, 0],
            label=MICNAME_MAP[mic_name],
            alpha=0.8,
            c=anglegram_rms,
            s=10,
            vmin=anglegram_db_threshold,
            vmax=np.nanmax(anglegram_rms),
        )
        error_dots = ax[0].scatter(
            timestamp,
            diff[:, 0],
            label="Modulo abs error",
            alpha=0.7,
            s=10,
            color="tab:orange",
        )
        ax[0].set_ylabel("Azimuth (rad)")
        ax[0].set_title(
            f"Error mean: {np.nanmean(diff[:, 0]):.2f}, std: {np.nanstd(diff[:, 0]):.2f}"
        )
        ax[0].legend(
            # handler_map={
            #     mocap_dots: HandlerNpoints(numpoints=1),
            #     anglegram_dots: HandlerNpoints(numpoints=1),
            #     error_dots: HandlerNpoints(numpoints=1),
            # },
            numpoints=3,
            loc="lower right",
        )
        yticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
        ax[0].set_yticks(yticks)
        for ytick in yticks:
            ax[0].axhline(y=ytick, linestyle="--", alpha=0.3)
        ax[0].set_ylim([-np.pi - 0.1, np.pi + 0.1])
        ax[0].set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

        ax[1].scatter(
            timestamp,
            mocap_polar[:, 1],
            label="Motion capture",
            alpha=0.8,
            s=10,
            color="tab:pink",
        )
        ax[1].scatter(
            timestamp,
            anglegram_polar[:, 1],
            label=MICNAME_MAP[mic_name],
            alpha=0.8,
            c=anglegram_rms,
            s=10,
            vmin=anglegram_db_threshold,
            vmax=np.nanmax(anglegram_rms),
        )
        ax[1].scatter(
            timestamp,
            diff[:, 1],
            label="Modulo abs error",
            alpha=0.7,
            s=10,
            color="tab:orange",
        )
        ax[1].set_ylabel("Elevation (rad)")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_title(
            f"Error mean: {np.nanmean(diff[:, 1]):.2f}, std: {np.nanstd(diff[:, 1]):.2f}"
        )
        ax[1].legend(loc="lower right")
        yticks = [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]
        ax[1].set_yticks(yticks)
        for ytick in yticks:
            ax[1].axhline(y=ytick, linestyle="--", alpha=0.3)
        ax[1].set_ylim([-np.pi / 2 - 0.1, np.pi / 2 + 0.1])
        ax[1].set_yticklabels(
            [r"$-\pi/2$", r"$-\pi/4$", r"$0$", r"$\pi/4$", r"$\pi/2$"]
        )

        cb = plt.colorbar(anglegram_dots, ax=ax.ravel().tolist())
        cb.ax.set_xlabel("Mic RMS\n(dB)")

        if save_path is not None:
            plt.savefig(save_path)
            print(f"=> Figure saved to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    anglegram_db_threshold = -35
    pairs = [
        [
            "../audio/trimmed/AEM/B_sync_1.5H_Mic1_230602_007.npz",
            "../Mocap/020623_AmbiREC/Data/Trimmed/trimmed_Movement0018.tsv",
            "Mic1",
            anglegram_db_threshold,
            "../figures/mocap_anglegram_th35db/B_sync_1.5H_Mic1_230602_007.png",
        ],
        [
            "../audio/trimmed/AEM/B_sync_1.5H_Mic2_230602_007.npz",
            "../Mocap/020623_AmbiREC/Data/Trimmed/trimmed_Movement0018.tsv",
            "Mic2",
            anglegram_db_threshold,
            "../figures/mocap_anglegram_th35db/B_sync_1.5H_Mic2_230602_007.png",
        ],
        [
            "../audio/trimmed/AEM/B_sync_1.5H_Mic3_230602_007.npz",
            "../Mocap/020623_AmbiREC/Data/Trimmed/trimmed_Movement0018.tsv",
            "Mic3",
            anglegram_db_threshold,
            "../figures/mocap_anglegram_th35db/B_sync_1.5H_Mic3_230602_007.png",
        ],
        [
            "../audio/trimmed/AEM/B_sync_1.5H_Mic4_230602_007.npz",
            "../Mocap/020623_AmbiREC/Data/Trimmed/trimmed_Movement0018.tsv",
            "Mic4",
            anglegram_db_threshold,
            "../figures/mocap_anglegram_th35db/B_sync_1.5H_Mic4_230602_007.png",
        ],
        [
            "../audio/trimmed/AEM/B_sync_1.5H_Mic5_230602_007.npz",
            "../Mocap/020623_AmbiREC/Data/Trimmed/trimmed_Movement0018.tsv",
            "Mic5",
            anglegram_db_threshold,
            "../figures/mocap_anglegram_th35db/B_sync_1.5H_Mic5_230602_007.png",
        ],
        [
            "../audio/trimmed/AEM/B_sync_Hby2_Mic1_230602_006.npz",
            "../Mocap/020623_AmbiREC/Data/Trimmed/trimmed_Movement0017.tsv",
            "Mic1",
            anglegram_db_threshold,
            "../figures/mocap_anglegram_th35db/B_sync_Hby2_Mic1_230602_006.png",
        ],
        [
            "../audio/trimmed/AEM/B_sync_Hby2_Mic2_230602_006.npz",
            "../Mocap/020623_AmbiREC/Data/Trimmed/trimmed_Movement0017.tsv",
            "Mic2",
            anglegram_db_threshold,
            "../figures/mocap_anglegram_th35db/B_sync_Hby2_Mic2_230602_006.png",
        ],
        [
            "../audio/trimmed/AEM/B_sync_Hby2_Mic3_230602_006.npz",
            "../Mocap/020623_AmbiREC/Data/Trimmed/trimmed_Movement0017.tsv",
            "Mic3",
            anglegram_db_threshold,
            "../figures/mocap_anglegram_th35db/B_sync_Hby2_Mic3_230602_006.png",
        ],
        [
            "../audio/trimmed/AEM/B_sync_Hby2_Mic4_230602_006.npz",
            "../Mocap/020623_AmbiREC/Data/Trimmed/trimmed_Movement0017.tsv",
            "Mic4",
            anglegram_db_threshold,
            "../figures/mocap_anglegram_th35db/B_sync_Hby2_Mic4_230602_006.png",
        ],
        [
            "../audio/trimmed/AEM/B_sync_Hby2_Mic5_230602_006.npz",
            "../Mocap/020623_AmbiREC/Data/Trimmed/trimmed_Movement0017.tsv",
            "Mic5",
            anglegram_db_threshold,
            "../figures/mocap_anglegram_th35db/B_sync_Hby2_Mic5_230602_006.png",
        ],
        [
            "../audio/trimmed/AEM/B_sync_H_Mic1_230602_005.npz",
            "../Mocap/020623_AmbiREC/Data/Trimmed/trimmed_Movement0016.tsv",
            "Mic1",
            anglegram_db_threshold,
            "../figures/mocap_anglegram_th35db/B_sync_H_Mic1_230602_005.png",
        ],
        [
            "../audio/trimmed/AEM/B_sync_H_Mic2_230602_005.npz",
            "../Mocap/020623_AmbiREC/Data/Trimmed/trimmed_Movement0016.tsv",
            "Mic2",
            anglegram_db_threshold,
            "../figures/mocap_anglegram_th35db/B_sync_H_Mic2_230602_005.png",
        ],
        [
            "../audio/trimmed/AEM/B_sync_H_Mic3_230602_005.npz",
            "../Mocap/020623_AmbiREC/Data/Trimmed/trimmed_Movement0016.tsv",
            "Mic3",
            anglegram_db_threshold,
            "../figures/mocap_anglegram_th35db/B_sync_H_Mic3_230602_005.png",
        ],
        [
            "../audio/trimmed/AEM/B_sync_H_Mic4_230602_005.npz",
            "../Mocap/020623_AmbiREC/Data/Trimmed/trimmed_Movement0016.tsv",
            "Mic4",
            anglegram_db_threshold,
            "../figures/mocap_anglegram_th35db/B_sync_H_Mic4_230602_005.png",
        ],
        [
            "../audio/trimmed/AEM/B_sync_H_Mic5_230602_005.npz",
            "../Mocap/020623_AmbiREC/Data/Trimmed/trimmed_Movement0016.tsv",
            "Mic5",
            anglegram_db_threshold,
            "../figures/mocap_anglegram_th35db/B_sync_H_Mic5_230602_005.png",
        ],
    ]

    for pair in tqdm(pairs):
        compute_and_plot_doae(*pair)
