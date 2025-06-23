import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401

NAME_CONVERT = {
    "Mic1": "FrontLeft",
    "Mic2": "FrontRight",
    "Mic3": "BackRight",
    "Mic4": "BackLeft",
    "Mic5": "Center",
}


def read_tsv(file_path):
    return pd.read_csv(file_path, skiprows=11, sep="\t")


def get_direction_to_mic(file_path, mic_name, coordinate="polar"):
    """
    Compute the direction of the sound source relative to the microphone in polar coordinate.

    args:
        file_path: str, path to the csv file
        mic_name: str, name of the microphone
        coordinate: str, "polar" or "cartesian"

    return:
        pd.DataFrame, a DataFrame with azimuth, elevation, and distance (in radiants)
    """
    assert mic_name in NAME_CONVERT, f"Invalid mic_name: {mic_name}"
    assert coordinate in ["polar", "cartesian"], f"Invalid coordinate: {coordinate}"

    df = read_tsv(file_path)

    # get XYZ position of the microphone
    mic_position = (
        df[
            [
                f"{NAME_CONVERT[mic_name]} X",
                f"{NAME_CONVERT[mic_name]} Y",
                f"{NAME_CONVERT[mic_name]} Z",
            ]
        ]
        .to_numpy()
        .mean(axis=0)
    )
    source_position = df[["Moving_D X", "Moving_D Y", "Moving_D Z"]].to_numpy()

    timestamp = df["Time"].to_numpy()

    if coordinate == "polar":
        return xyz_to_polar(source_position - mic_position), timestamp
    elif coordinate == "cartesian":
        return source_position - mic_position, timestamp


def xyz_to_polar(xyz, zero_azimuth="north", pos_direction="cc"):
    """
    Convert the XYZ coordinate to polar coordinate.

    args:
        xyz: np.array
            shape (N, 3), XYZ coordinate
        zero_azimuth: str
            azimuth angle of the zero degree, default is "north"
        pos_direction: "cc" or "cw"
            positive direction of the azimuth angle, "cc" for counter-clockwise and "cw" for clockwise.

    return:
        np.array, shape (N, 3), polar coordinate (azimuth, elevation, distance)
    """
    assert zero_azimuth == "north", "Only support zero_azimuth = 'north'"

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    distance = np.sqrt(x**2 + y**2 + z**2)
    if pos_direction == "cc":
        azimuth = -np.arctan2(x, y)  # azimuth = arccot y/x = arctan x/y
    elif pos_direction == "cw":
        azimuth = np.arctan2(x, y)
    elevation = np.arcsin(z / distance)

    return np.stack([azimuth, elevation, distance], axis=1)


def plot_mocap_polar(file_path, mic_name, save_path=None):
    """
    Plot the polar coordinate of the sound source.

    """
    df = read_tsv(file_path)
    coords = get_direction_to_mic(file_path, mic_name)
    azimuth, elevation, distance = coords[:, 0], coords[:, 1], coords[:, 2]

    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(3, 1, figsize=(9, 6))
        fig.suptitle(f"{os.path.basename(file_path)}, {mic_name}", fontsize=16)

        ax[0].plot(df["Time"], azimuth)
        ax[0].set_ylabel("Azimuth (rad)")
        ax[0].set_ylim([-np.pi - 0.1, np.pi + 0.1])
        ax[0].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax[0].set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

        ax[1].plot(df["Time"], elevation)
        ax[1].set_ylabel("Elevation (rad)")
        ax[1].set_ylim([-np.pi / 2 - 0.1, np.pi / 2 + 0.1])
        ax[1].set_yticks([-np.pi / 2, 0, np.pi / 2])
        ax[1].set_yticklabels([r"$-\pi/2$", r"$0$", r"$\pi/2$"])

        ax[2].plot(df["Time"], distance)
        ax[2].set_xlabel("Time (s)")
        ax[2].set_ylabel("Distance (mm)")

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


def plot_mocap(file_path, save_path=None):
    df = read_tsv(file_path)

    names = ["Moving_D", "Center", "FrontLeft", "FrontRight", "BackLeft", "BackRight"]

    with plt.style.context(["science", "no-latex"]):
        ax = plt.figure(figsize=(10, 10)).add_subplot(projection="3d")

        for name in names:
            X, Y, Z = (
                df[f"{name} X"].to_numpy(),
                df[f"{name} Y"].to_numpy(),
                df[f"{name} Z"].to_numpy(),
            )

            if "Moving" in name:
                ax.plot(X, Y, Z)  # Plot contour curves
            else:
                ax.scatter(X, Y, Z)
            ax.set(zlim=(0, 2000))

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
