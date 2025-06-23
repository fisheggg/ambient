import numpy as np
import torch
from librosa import amplitude_to_db
from librosa.util import frame
from tqdm import tqdm

from .common import AmbiFormat
from .decoder import AmbiDecoder
from .position import Position


def spherical_mesh(angular_res=None, n_phi=None, n_nu=None):
    """
    returns two meshes for phi and nu
    shape: (n_nu, n_phi)
    first element is (-pi/2, -pi)
    """
    if angular_res is None and n_phi is None and n_nu is None:
        raise ValueError("Either angular_res or n_phi and n_nu must be provided")
    elif angular_res is not None:
        phi_rg = np.arange(-180.0, 180.0, angular_res) / 180.0 * np.pi
        nu_rg = np.arange(-90.0, 90.0, angular_res) / 180.0 * np.pi
    elif n_phi is not None and n_nu is not None:
        phi_rg = np.linspace(-np.pi, np.pi, n_phi)
        nu_rg = np.linspace(-np.pi / 2, np.pi / 2, n_nu)
    elif n_phi is None or n_nu is None:
        raise ValueError("Both n_phi and n_nu must be provided")

    phi_mesh, nu_mesh = np.meshgrid(phi_rg, nu_rg)
    return phi_mesh, nu_mesh


class AEMGenerator(object):
    def __init__(
        self,
        frame_length: int,
        hop_length: int,
        ambi_order: int = 1,
        angular_res: float = None,
        n_phi: int = None,
        n_nu: int = None,
        batch_size: int = None,
        gpu: bool = False,
        show_progress: bool = False,
    ):
        """
        generates audio energy maps from ambisonics audio

        Args:
            frame_length: int, frame length in samples
            hop_length: int, hop length in samples
            ambi_order: int, ambisonics order
            window: int, window size in samples
            angular_res: float, angular resolution in degrees. either angular_res or n_phi and n_nu must be provided
            n_phi: int, number of phi bins
            n_nu: int, number of nu bins
            gpu: bool, use GPU for decoding
            batch_size: int, batch size for decoding
            show_progress: bool, show progress bar
        """
        self.angular_res = angular_res
        self.phi_mesh, self.nu_mesh = spherical_mesh(angular_res, n_phi, n_nu)
        self.frame_shape = (n_nu, n_phi)
        self.n_speakers = n_nu * n_phi
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.gpu = gpu
        self.batch_size = batch_size
        self.show_progress = show_progress
        mesh_p = [
            Position(phi, nu, 1.0, "polar")
            for phi, nu in zip(self.phi_mesh.reshape(-1), self.nu_mesh.reshape(-1))
        ]

        # Setup decoder
        self.decoder = AmbiDecoder(
            mesh_p, AmbiFormat(ambi_order), method="projection", gpu=gpu
        )

    def compute(self, audio: np.ndarray):
        """
        audio: (n_samples, n_harmonics)
        """
        assert len(audio.shape) == 2

        # frame the audio into ((n_batchs), n_frames, self.frame_length, n_harmonics)
        data = frame(
            audio, frame_length=self.frame_length, hop_length=self.hop_length, axis=0
        )
        n_frames = data.shape[0]
        # data = torch.from_numpy(data)
        if self.batch_size is not None:
            # pad the data to make it divisible by batch_size
            n_pad = self.batch_size - (data.shape[0] % self.batch_size)
            data = np.pad(data, ((0, n_pad), (0, 0), (0, 0)), mode="constant")
            data = frame(
                data, frame_length=self.batch_size, hop_length=self.batch_size, axis=0
            )
        # if self.gpu:
        # data = torch.from_numpy(data).cuda()
        # print(f"Data shape: {data.shape}")

        # decoded final shape: (n_frames, n_speakers)
        if self.batch_size is not None:
            # decoded init shape: ((n_batchs), n_frames, frame_length, n_speakers)
            decoded = np.zeros((data.shape[0], self.batch_size, self.n_speakers))
            # if self.gpu:
            # decoded = torch.from_numpy(decoded).cuda()
            range_ = range(data.shape[0])
            if self.show_progress:
                range_ = tqdm(range_, desc="Decoding batches")

            # print(f"=> memory before decoding: {torch.cuda.memory_summary()}")
            # actual decoding and rms computation
            for i in range_:
                data_in = torch.from_numpy(data[i, :, :, :]).cuda()
                decoded[i, :, :] = np.sqrt(
                    np.mean(self.decoder.decode(data_in).cpu().numpy() ** 2, axis=1)
                )
                del data_in
                # print(f"=> batch idx: {i}, memory: {torch.cuda.memory_summary()}")

            # reshape to (n_frames, n_phi, n_nu)
            decoded = decoded.reshape(-1, *self.frame_shape)[:n_frames]

        else:
            decoded = self.decoder.decode(data)
            # compute rms
            decoded = np.sqrt(decoded.mean(decoded**2, axis=1))
            # reshape to (n_frames, n_phi, n_nu)
            decoded = decoded.reshape(decoded.shape[0], *self.frame_shape)

        # print(f"Decoded shape: {decoded.shape}")
        # print(f"RMS shape: {rms.shape}")
        return amplitude_to_db(decoded)
