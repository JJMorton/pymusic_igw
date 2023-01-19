#!/usr/bin/env python3

from os import path
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from numpy import float64, int64
from numpy.typing import NDArray
from typing import List, Tuple
import logging

from tomso import fgong

import pygyre

from pymusic.spec import SphericalHarmonicsTransform1D
from pymusic.spec import NuFFT1D
from pymusic.spec import BlackmanWindow
from pymusic.big_array import BigArray
from pymusic.big_array import SphHarm1DArray
from pymusic.big_array import FFTPowerSpectrumArray
from pymusic.math import SphericalMidpointQuad1D

from analysis_task import AnalysisTask


# Set up logging
logger = logging.getLogger(__name__)


def log_freq_diff(desired: NDArray[float64], have: NDArray[float64], percent_warn: float = 0.5):
    percent_diff = 100 * np.abs(have - desired) / desired
    logger.info("Found frequencies of: " + str(have) + ", wanted " + str(desired))
    if (percent_diff > percent_warn).any():
        logger.warning("Percentage difference of freqs from those requested: " + str(percent_diff))


def compute_PS(
    sim_data: BigArray,
    quad: SphericalMidpointQuad1D,
    field_name: str,
    mode_ells: NDArray[int64],
    mode_freqs: NDArray[float64]
    ) -> NDArray[float64]:

    logger.info("Computing power spectrum")

    # Setup FFT transforms
    dt = np.mean(np.diff(np.array(sim_data.labels_along_axis("time"))))
    fft = NuFFT1D(window=BlackmanWindow(), sampling_period=dt, spacing_tol=0.1)

    # Setup spherical harmonics
    sh_xform = SphericalHarmonicsTransform1D(quad, ell_max=np.max(mode_ells))

    # Calculate PS of radial velocity, has shape (omega, r, mode_ells)
    logger.info("Getting radial velocity field")
    field = sim_data.xs(field_name, axis="var")
    logger.info("Constructing PS pipeline")
    PS = FFTPowerSpectrumArray(
        SphHarm1DArray(
            field,
            sh_xform,
            theta_axis="x2",
            ell_axis="ell",
            ells=mode_ells.tolist(),
        ),
        fft,
        "time",
        "freq",
    ).scaled(1.0 / (4 * np.pi))

    # Pick some frequencies to slice at
    logger.info("Evaluating frequency axis")
    freqs = np.array(PS.labels_along_axis("freq"))
    logger.info("Finding frequencies to sample at")
    adjusted_freq_idx = np.array([np.abs(freqs - f).argmin() for f in mode_freqs])
    adjusted_freq = freqs[adjusted_freq_idx]
    log_freq_diff(mode_freqs, adjusted_freq)

    # PS_selected contains the PS for the specified (ell, omega)
    # Of shape (no. of modes, radius)
    logger.info("Evaluating selected modes")
    PS_selected = np.array(
        [
            # PS.xs(ell, "ell").take([freqs[i - 1], freqs[i], freqs[i + 1]], "freq").mean("freq").array()
            # for i, ell in zip(adjusted_freq_idx, mode_ells)
            PS.slabbed("x1", 256).xs(ell, "ell").xs(f, "freq").array()
            for f, ell in zip(adjusted_freq, mode_ells)
        ]
    )

    return PS_selected


def get_gyre_displacement(
    radii: NDArray[float64], details_dir: str, ell: int, freq: float
    ) -> Tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:

    xi_r = xi_h = xi_hypot = x = np.array([])
    paths = glob(path.join(details_dir, f"detail*.l{ell}*.h5"))
    if len(paths) == 0:
        logging.error(f"No gyre detail files found for ell={ell}")
        return xi_r, xi_h, xi_hypot

    mindiff = freq
    freq_match = 0.0
    for file in paths:
        details = pygyre.read_output(file)
        diff = np.abs(details.meta['freq'].real - freq)
        if diff > mindiff:
            continue
        mindiff = diff
        freq_match = details.meta['freq'].real
        x = details.columns.get('x').data
        xi_r = details.columns.get('xi_r').data.real
        xi_h = details.columns.get('xi_h').data.real

    log_freq_diff(np.array([freq]), np.array([freq_match]))

    xi_r = np.interp(radii, x, xi_r)
    xi_h = np.interp(radii, x, xi_h)
    xi_hypot = np.hypot(xi_r, xi_h)
    return xi_r, xi_h, xi_hypot


class LuminosityModes(AnalysisTask):

    def __init__(self):
        super().__init__("luminosity_modes")

    def compute(self):
        # Some of the modes found by GYRE that we want to compare to MUSIC
        modes: List[Tuple[int, float]] = [
            # UNNO outer BC
            # (Angular order ell, frequency omega)
            # (2, 3.804e-6),
            # (3, 7.301e-6),
            # (4, 9.369e-6),
            # (7, 19.596e-6),

            # DZIEM outer BC
            # (Angular order ell, frequency omega)
            # (2, 3.981e-6),
            # (3, 7.750e-6),
            # (4, 9.936e-6),
            # (7, 18.286e-6),

            # GAMMA outer BC
            # (Angular order ell, frequency omega)
            (2, 3.816430447442817e-6),
            (3, 7.348460794932171e-6),
            (4, 9.438993003628058e-6),
            (7, 15.44425232235460e-6),
        ]
        selected_ell = np.array([m[0] for m in modes])
        selected_freq = np.array([m[1] for m in modes])

        # Setup simulation from dump files
        times = self.sim_data.labels_along_axis("time")
        radii = np.array(self.sim_data.labels_along_axis("x1"))
        PS = compute_PS(self.sim_data, self.quad, "vel_1", selected_ell, selected_freq)

        return (
            self.dump_files,
            modes,
            radii,
            times,
            PS
        )

    def plot(self, result) -> plt.Figure:
        (dump_files, modes, radii, times, PS) = result

        # The radii to plot between (units of stellar radii)
        radius_domain = (0.25, 1)

        # Read displacements from GYRE
        logger.info("Reading GYRE detail files")
        xi = [ get_gyre_displacement(radii / self.params.radius, path.join(self.base_dir, "gyre_detail/"), ell, freq * 1e6)[2] for ell, freq in modes ]

        # Load data from 1D profile
        filename_fgong = path.join(self.base_dir, "fort11.gong.z2m20_krad_mesa_AS1d12_mod1420")
        logger.info("Loading data from 1D profile (FGONG format)")
        profile_fgong = fgong.load_fgong(filename_fgong)
        r_fgong = np.flip(profile_fgong.r)
        BVF_fgong = np.flip(np.sqrt(np.abs(profile_fgong.N2)) / (2 * np.pi))
        rho_fgong = np.flip(profile_fgong.rho)
        kh_fgong = 1 / (5.68 * r_fgong)  # sum_ell 1/ell ~ 5.68 for ell = 1 to 200
        # Interpolate 1D quantities onto the same r axis as MUSIC
        BVF = np.interp(radii, r_fgong, BVF_fgong)
        rho = np.interp(radii, r_fgong, rho_fgong)
        kh = np.interp(radii, r_fgong, kh_fgong)

        # Finally, calculate wave luminosity
        wave_lum = (
            4
            * np.pi
            * radii[np.newaxis, :] ** 2
            * (PS * rho[np.newaxis, :] * BVF[np.newaxis, :])
            / (2 * kh[np.newaxis, :])
        )
        
        # Trim result to selected radii
        radii = radii / self.params.radius
        radius_domain_filter = np.where(np.logical_and(radii > radius_domain[0], radii < radius_domain[1]))
        radii = radii[radius_domain_filter]
        wave_lum = [ lum[radius_domain_filter] for lum in wave_lum ]
        xi = [ x[radius_domain_filter] for x in xi ]

        # Plot everything!
        nummodes = len(modes)
        numcols = 2
        numrows = int(np.ceil(nummodes / numcols))
        fig = plt.figure(figsize=(numcols * 4, numrows * 3))
        fig.suptitle(r"t={:.3E} to {:.3E} ({} dumps)".format(times[0], times[-1], len(dump_files)))

        for i in range(nummodes):
            ax = fig.add_subplot(numrows, numcols, i + 1)

            # Plot Lwave from MUSIC
            ax.set_title(r"$\omega={:.3f}$$\mu$Hz $\ell={}$".format(modes[i][1] * 1e6, modes[i][0]))
            ax.plot(radii, wave_lum[i], color=f"C{i}", lw=2)
            ax.set_xlabel(r"$r/R$")
            ax.set_ylabel(r"$L_{\rm wave}$")
            ax.set_yscale("log")

            # Plot |xi| from GYRE
            ax_twin = ax.twinx()
            ax_twin.plot(radii, xi[i], c="grey", lw=1, alpha=0.8)
            ax_twin.set_ylabel(r"$|\xi|$ from GYRE")
            ax_twin.set_yticks([])
            ax_twin.set_yticklabels([])
            ax_twin.set_yscale("log")

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

if __name__ == "__main__":
    LuminosityModes().run()
