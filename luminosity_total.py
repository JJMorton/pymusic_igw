import matplotlib.pyplot as plt
import numpy as np
from os import path

from tomso import fgong

from pymusic.spec import NuFFT1D
from pymusic.spec import BlackmanWindow

from pymusic.big_array import FFTPowerSpectrumArray
from pymusic.big_array import CachedArray

from analysis_task import AnalysisTask


class LuminosityTotal(AnalysisTask):

    def __init__(self):
        super().__init__("luminosity_total")

    def compute(self):

        # Setup simulation from dump files
        radii = np.array(self.sim_data.labels_along_axis("x1"))

        # Setup FFT transforms
        dt = np.mean(np.diff(np.array(self.sim_data.labels_along_axis("time"))))
        times = self.sim_data.labels_along_axis("time")
        fft = NuFFT1D(window=BlackmanWindow(), sampling_period=dt, spacing_tol=0.1)

        # Calculate PS of radial velocity
        # PS is averaged in angular direction with 'quad.average'
        vel_r = self.sim_data.xs("vel_1", axis="var")
        PS = CachedArray(
            FFTPowerSpectrumArray(
                vel_r,
                fft,
                "time",
                "freq",
            )
            .collapse(self.quad.average, "x2")
            # .slabbed("x1", 64)
            .scaled(1.0 / (4 * np.pi))
        )

        # Pick some frequencies to slice at
        freqs = np.array(PS.labels_along_axis("freq"))
        selected_freq = [2e-6, 5e-6, 10e-6, 20e-6, 30e-6]
        selected_freq_idx = [np.abs(freqs - f).argmin() for f in selected_freq]

        # PSD_norm is the power spectrum and has shape (selected_freqs, rads)
        PSD_norm = []
        for ff in selected_freq_idx:
            # average the PS over three frequency bins
            spect = PS.take(freqs[ff - 1 : ff + 2], "freq").mean("freq").array()
            PSD_norm.append(spect)


        return times, selected_freq, radii, PSD_norm

    def plot(self, result):
        (times, selected_freq, radii, PSD_norm) = result

        # Load data from 1D profile
        FILENAME_GYRE = path.join(self.base_dir, "fort11.gong.z2m20_krad_mesa_AS1d12_mod1420")
        print("load data from 1D profile (FGONG format)")
        profile_fgong = fgong.load_fgong(FILENAME_GYRE)
        r_fgong = np.flip(profile_fgong.r)
        BVF_fgong = np.flip(np.sqrt(np.abs(profile_fgong.N2)) / (2 * np.pi))
        rho_fgong = np.flip(profile_fgong.rho)
        kh_fgong = 1 / (5.68 * r_fgong)  # sum_ell 1/ell ~ 5.68 for ell = 1 to 200
        # Interpolate quantities onto the same r axis as MUSIC
        BVF = np.interp(radii, r_fgong, BVF_fgong)
        rho = np.interp(radii, r_fgong, rho_fgong)
        kh = np.interp(radii, r_fgong, kh_fgong)

        # Calculate wave luminosity
        wave_lum = (
            4
            * np.pi
            * radii[np.newaxis, :] ** 2
            * (PSD_norm * rho[np.newaxis, :] * BVF[np.newaxis, :])
            / (2 * kh[np.newaxis, :])
        )

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(r"t={:.3E} to {:.3E}".format(times[0], times[-1]))
        for i in range(len(selected_freq)):
            label = r"$\omega = {}$ $\mu$Hz".format(selected_freq[i] * 1e6)
            ax.plot(radii / self.params.radius, wave_lum[i], label=label)
        ax.set_xlabel(r"$r/R$")
        ax.set_ylabel(r"$L$")
        ax.set_yscale("log")
        ax_twin = ax.twinx()
        ax_twin.plot(radii / self.params.radius, BVF * 1e6, ls="--", c="grey")
        ax_twin.set_ylabel(r"$N$ ($\mu$Hz)")
        ax.legend()
        return fig


if __name__ == "__main__":
    LuminosityTotal().run()
