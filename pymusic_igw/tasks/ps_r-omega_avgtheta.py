#!/usr/bin/env python3

from os import path
import matplotlib.pyplot as plt
import numpy as np
import logging
import matplotlib

from tomso import fgong

from pymusic.spec import SphericalHarmonicsTransform1D
from pymusic.spec import NuFFT1D
from pymusic.spec import BlackmanWindow
from pymusic.big_array import TimedArray
from pymusic.big_array import SphHarm1DArray
from pymusic.big_array import FFTPowerSpectrumArray

from pymusic_igw import AnalysisTask


# Set up logging
logger = logging.getLogger(__name__)


class PowerSpecAvgTheta(AnalysisTask):

    def __init__(self):
        super().__init__("ps_r-omega_avgtheta")

    def compute(self):
        field = "vel_1"
        times = np.array(self.sim_data.labels_along_axis("time"))
        logger.info(f"Times from t={times[0]:E} to t={times[-1]:E}")
        dt = np.mean(np.diff(times))
        fft = NuFFT1D(window=BlackmanWindow(), sampling_period=dt, spacing_tol=0.07)

        radii = np.array(self.sim_data.labels_along_axis("x1"))
        radii_toplot = np.arange(self.params.boundary_conv, 0.38 * self.params.radius, 0.02 * self.params.radius)
        radii_idx = [np.abs(radii - r).argmin() for r in radii_toplot]
        radii = radii[radii_idx]

        logger.info("Setting up power spectrum pipeline")
        # spec.shape ~ (freq, r)
        spec = TimedArray(
            FFTPowerSpectrumArray(
                self.sim_data.xs(field, axis="var"),
                fft,
                "time",
                "freq",
            )
            .take(radii, axis="x1")
            .collapse(self.quad.average, "x2")
            # .slabbed("x1", 256)
            .scaled(1.0 / (4 * np.pi))
        )

        freqs = np.array(spec.labels_along_axis("freq"))
        radii = np.array(spec.labels_along_axis("x1"))

        logger.info("Computing selected power spectra")
        return (field, radii, freqs, spec.array())

    def plot(self, result):
        field, radii, freqs, spec = result
        print(radii.shape, freqs.shape, spec.shape)

        # # Plot single frequencies
        # selected_freqs = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]) * 1e-6
        # idx = [np.abs(freqs - f).argmin() for f in selected_freqs]
        # fig = plt.figure()
        # fig.suptitle(f"Power Spectrum for Field '{field}'")
        # ax = fig.add_subplot(1, 1, 1)
        # for i in idx:
        #     ax.plot(radii, spec[i], label=r"$\omega = {:.2E}\mu \rm Hz$".format(freqs[i] * 1e6))
        # ax.set_xlabel(r"$r$")
        # ax.set_ylabel(r"$P[v_r]$")
        # fig.legend()
        # ax.set_yscale("log")

        # Load data from 1D profile
        filename_fgong = path.join(self.base_dir, "fort11.gong.z2m20_krad_mesa_AS1d12_mod1420")
        profile_fgong = fgong.load_fgong(filename_fgong)
        r_fgong = np.flip(profile_fgong.r)
        BVF_fgong = np.flip(np.sqrt(np.abs(profile_fgong.N2)) / (2 * np.pi))
        # Interpolate 1D quantities onto the same r axis as MUSIC
        BVF = np.interp(radii, r_fgong, BVF_fgong)

        # Plot single radii
        # radii_toplot = np.arange(0, 1, 0.25) * (radii[-1] - self.params.boundary_conv) + self.params.boundary_conv
        radii_toplot = np.arange(self.params.boundary_conv, 0.38 * self.params.radius, 0.02 * self.params.radius)
        radii_idx = [np.abs(radii - r).argmin() for r in radii_toplot]
        radii = radii[radii_idx]
        spec = spec.T[radii_idx]
        # Now, spec is of shape (radii, freq)
        col = plt.get_cmap("coolwarm")
        norm = matplotlib.colors.Normalize(vmin=radii[0], vmax=radii[-1])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for spec_r, r in zip(spec, radii):
            ax.plot(freqs*1e6, spec_r, c=col(norm(r)), label=r"$r = {:.3f}R_*$".format(r/self.params.radius))
        ax.set_xlim(left=freqs[1] * 1e6, right=0.5/1e3 * 1e6)
        ax.set_ylim(top=1e11, bottom=1e-5)
        ax.set_xlabel(r"$\omega$ ($\mu$Hz)")
        ax.set_ylabel(r"$P[v_r]$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(x=1 / self.params.tau_conv * 1e6, c="blue", lw=1)
        # ax.axvline(x=0.5 / 9e3 * 1e6, c="black", lw=1, ls='--')
        ax.grid()
        ax.legend(loc=3)

        # # Plot a colourplot of every frequency
        # fig = plt.figure(figsize=(7, 6))
        # fig.suptitle(f"Power Spectrum for Field '{field}'")
        # ax = fig.add_subplot(1, 1, 1)
        # mesh = ax.pcolormesh(radii, freqs*1e6, spec, cmap='inferno', norm=matplotlib.colors.LogNorm(vmin=1e0, vmax=1e12))
        # # ax.plot(radii, BVF*1e6, c="white", ls='--', label="BVF (full domain)")
        # fig.colorbar(mesh)
        # ax.set_xlabel(r"$r$")
        # ax.set_ylabel(r"$\omega$ ($\mu$Hz)")
        # # ax.legend()

        fig.tight_layout(rect=[0, 0.02, 1, 0.98])
        return fig

if __name__ == "__main__":
    PowerSpecAvgTheta().run()
